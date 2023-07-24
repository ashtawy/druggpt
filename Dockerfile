# app/Dockerfile

FROM python:3.7-slim

WORKDIR /app

RUN apt-get update && apt-get install -y \
    build-essential \
    curl \
    software-properties-common \
    libxrender1 \
    git \
    && rm -rf /var/lib/apt/lists/*
ADD . /app
# RUN git clone https://github.com/streamlit/streamlit-example.git .

# RUN pip3 install -r requirements.txt --extra-index-url https://download.pytorch.org/whl/cu117

RUN pip3 install -e .

EXPOSE 2026

HEALTHCHECK CMD curl --fail http://localhost:2026/_stcore/health