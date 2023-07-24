import argparse
import csv
import hashlib
import os
import platform
import shutil
import subprocess
import sys
import tempfile
import warnings

import mols2grid
import numpy as np
import pandas as pd
import streamlit as st
import streamlit.components.v1 as components
import torch
from rdkit import Chem
from tqdm import tqdm
from transformers import AutoTokenizer, GPT2LMHeadModel


def display_results(df_result, smiles_column, key):
    clst_ui_clms = st.columns(2)
    subset_options = [c for c in df_result.columns if c != smiles_column] + ["img"]
    subset = clst_ui_clms[0].multiselect(
        "columns to show on grid",
        subset_options,
        default=["img"],
        key=f"{key}_subset",
    )
    tooltip = clst_ui_clms[1].multiselect(
        "Columns to show as tooltips",
        df_result.columns.tolist(),
        key=f"{key}_tooltip",
    )
    grid_clms = st.columns(2)
    n_cols = grid_clms[0].number_input(
        "Number of columns",
        min_value=1,
        max_value=20,
        value=6,
        key=f"{key}_n_cols",
    )
    n_rows = grid_clms[1].number_input(
        "Number of rows",
        min_value=1,
        max_value=10,
        value=3,
        key=f"{key}_n_rows",
    )

    # if st.session_state["n_cols"] is None:
    #   st.session_state["n_cols"] = 6
    # if st.session_state["n_rows"] is None:
    #    st.session_state["n_rows"] = 3

    raw_html = mols2grid.display(
        df_result,
        smiles_col=smiles_column,
        subset=subset,
        tooltip=tooltip,
        n_cols=n_cols,
        n_rows=n_rows,
        # template="table",
        # prerender=True,
    )._repr_html_()
    height = 600 if df_result.shape[0] > 16 else 400
    height = height + len(subset) * 10
    width = 165 * n_cols
    components.html(raw_html, width=width, height=height, scrolling=True)


# Function to read in FASTA file
def parse_fasta_file(file):
    sequence = []

    for line in file:
        line = line.decode("utf-8").strip()
        if not line.startswith(">"):
            sequence.append(line)

    protein_sequence = "".join(sequence)
    return protein_sequence


# What EGFR mutations are linked to cancer and at what location in the gene they appear


def get_data():

    uploaded_file = st.file_uploader(
        "1: Upload a FASTA file",
        type=["fasta"],
        accept_multiple_files=False,
        key="fasta_file_uploader",
    )
    if uploaded_file is not None:
        st.session_state["protein_sequence"] = parse_fasta_file(uploaded_file)

    seq = st.text_area("2: Or input a protein amino acid sequence.", key="text")
    if seq:
        st.session_state["protein_sequence"] = (
            seq.strip().replace("\n", "").replace(" ", "")
        )


def generate():
    if platform.system() == "Linux":
        os.environ["TOKENIZERS_PARALLELISM"] = "false"
    # Load the tokenizer and the model
    tokenizer = AutoTokenizer.from_pretrained("liyuesen/druggpt")
    model = GPT2LMHeadModel.from_pretrained("liyuesen/druggpt")

    model.eval()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    # Create a LigandPostprocessor object
    if st.session_state["directly_gen"]:
        prompt = "<|startoftext|><P>"
    elif st.session_state.get("protein_sequence") is not None:
        p_prompt = "<|startoftext|><P>" + st.session_state["protein_sequence"] + "<L>"
        l_prompt = "" + st.session_state["ligand_prompt"]
        prompt = p_prompt + l_prompt
    else:
        st.error("No protein sequence or ligand prompt provided.")
        return
    # Generate molecules
    generated = torch.tensor(tokenizer.encode(prompt)).unsqueeze(0)
    generated = generated.to(device)

    batch_number = 0

    directly_gen_protein_list = []
    directly_gen_ligand_list = []
    generate_ligand_list = []
    generate_ikeys_list = []
    prog_bar = st.progress(0, text="")
    info_bar = st.empty()
    while len(generate_ikeys_list) < st.session_state["num_generated"]:
        batch_number += 1
        info_bar.info(f"Batch {batch_number}: Generating ligand SMILES ...")
        sample_outputs = model.generate(
            generated,
            # bos_token_id=random.randint(1,30000),
            do_sample=True,
            top_k=st.session_state["top_k"],
            max_length=1024,
            top_p=st.session_state["top_p"],
            num_return_sequences=st.session_state["batch_generated_size"],
        )
        for sample_output in sample_outputs:
            generate_ligand = tokenizer.decode(
                sample_output, skip_special_tokens=True
            ).split("<L>")[1]
            try:
                mol = Chem.MolFromSmiles(generate_ligand)
                # generate inchi_key for the generated ligand
                inchi_key = Chem.inchi.MolToInchiKey(mol)
                if inchi_key not in generate_ikeys_list:
                    can_smiles = Chem.MolToSmiles(mol)
                    generate_ligand_list.append(can_smiles)
                    generate_ikeys_list.append(inchi_key)
                    if st.session_state["directly_gen"]:
                        directly_gen_protein_list.append(
                            tokenizer.decode(
                                sample_output, skip_special_tokens=True
                            ).split("<L>")[0]
                        )
                        directly_gen_ligand_list.append(can_smiles)
            except Exception as e:
                print(f"Unable to decode SMILES: {sample_output} ", e)
        n = len(generate_ikeys_list)
        N = st.session_state["num_generated"]
        prog_bar.progress(
            min(n / N, 1.0), f"Generated {n}/{N} compounds ({100*n/N:.2f}%)"
        )
        torch.cuda.empty_cache()
    st.session_state["ligands"] = pd.DataFrame(
        {"inchi_key": generate_ikeys_list, "smiles": generate_ligand_list}
    )


def get_configs():
    with st.expander("Configurations"):
        clms = st.columns(2)
        clms[0].text_input("Ligand prompt", value="", key="ligand_prompt")
        clms[1].number_input(
            "Number of ligands to generate", value=50, key="num_generated"
        )
        clms = st.columns(2)
        clms[0].number_input("top_k", value=5, key="top_k")
        clms[1].number_input("top_p", value=0.6, key="top_p")
        clms = st.columns(2)
        clms[0].number_input(
            "batch_generated_size", value=32, key="batch_generated_size"
        )
        st.session_state["directly_gen"] = clms[1].checkbox("Directly generate", False)


def main():
    st.set_page_config(layout="wide")
    st.header("DrugGPT")
    st.subheader("A generative drug design model based on GPT2")
    # search()
    get_data()
    get_configs()
    configs = {
        "ligand_prompt": "",
        "directly_generate": False,
        "num_generated": 50,
        "device": "cuda",
        "batch_generated_size": 32,
        "top_k": 5,
        "top_p": 0.6,
    }
    # question = st.text_input("Enter a question", key="question")
    if st.button("Generate"):
        if st.session_state.get("protein_sequence") is not None or st.session_state.get(
            "directly_gen"
        ):
            generate()
        else:
            st.error(
                "Must provide a protein sequence or ligand prompt or directly generate."
            )
    if st.session_state.get("ligands") is not None and isinstance(
        st.session_state.get("ligands"), pd.DataFrame
    ):
        display_results(
            st.session_state["ligands"].copy(), smiles_column="smiles", key="results"
        )

        st.download_button(
            label="Download all Enamine data",
            data=st.session_state["ligands"].to_csv(index=False).encode("utf-8"),
            file_name="gp2_generated_ligands.csv",
            mime="text/csv",
        )


if __name__ == "__main__":
    main()
