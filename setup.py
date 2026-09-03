"""Utilities for setup."""

from setuptools import setup

setup(
    name="fastcoref",
    description="Tapestry's inference-only FastCoref runtime for Transformers 5",
    version="3.0.0",
    license="MIT",
    author="Shon Otmazgin, Arie Cattan, Yoav Goldberg",
    author_email="shon711@gmail.com",
    packages=[
        "fastcoref",
        "fastcoref.coref_models",
        "fastcoref.utilities",
    ],
    url="https://github.com/shon-otmazgin/fastcoref",
    install_requires=[
        "tqdm>=4.64.0",
        "numpy>=1.21.6",
        "scipy>=1.7.3",
        "spacy>=3.0.6",
        "torch>=1.10.0",
        "transformers>=5.0.0",
        "datasets>=2.5.2",
    ],
)
