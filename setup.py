#!/usr/bin/env python
import io
import os

"""The setup script."""

from setuptools import find_packages, setup


def read(*paths, **kwargs):
    """Read the contents of a text file safely.
    >>> read("charlie", "VERSION")
    '0.1.0'
    >>> read("README.md")
    ...
    """

    content = ""
    with io.open(
        os.path.join(os.path.dirname(__file__), *paths),
        encoding=kwargs.get("encoding", "utf8"),
    ) as open_file:
        content = open_file.read().strip()
    return content


def read_requirements(path):
    return [
        line.strip()
        for line in read(path).split("\n")
        if not line.startswith(('"', "#", "-", "git+"))
    ]


test_requirements = [
    "pytest>=3",
]

setup(
    author="Hossam Ashtawy",
    author_email="hashtawy@1859.ai",
    python_requires=">=3.6",
    classifiers=[
        "Development Status :: 2 - Pre-Alpha",
        "Intended Audience :: Developers",
        "Natural Language :: English",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.6",
        "Programming Language :: Python :: 3.7",
        "Programming Language :: Python :: 3.8",
    ],
    description="A generative drug design model based on GPT2",
    install_requires=read_requirements("requirements.txt"),
    include_package_data=True,
    keywords="druggpt",
    name="druggpt",
    packages=find_packages(exclude=["tests", ".github"]),
    test_suite="tests",
    tests_require=test_requirements,
    url="https://github.com/hashtawy/druggpt",
    version="0.0.1",
    zip_safe=False,
)
