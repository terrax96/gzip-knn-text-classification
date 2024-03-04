# gzip-knn-text-classification

Implementation of the [“Low-Resource” TextClassification: A Parameter-Free Classification Method with Compressors](https://aclanthology.org/2023.findings-acl.426.pdf) paper.

The main idea is based on the fact that:
1. compressors are good at capturing regularities
1. objects from the same category share more regularities than those from different categories
1. the gzip compressed length can be used as an approximation of the Kolmogorov complexity (i.e. length of the shortest binary program that can generate the considered data)
1. the compressed length can be used to compute the Normalized Compression Distance (NCD)
1. NCD can be used as a similarity measure to define a kNN model to classify text

## Setup

- `python -m venv venv`
- Linux: `source venv/bin/activate` OR Windows: `Set-ExecutionPolicy Unrestricted -Scope Process; .\venv\Scripts\Activate.ps1`
- `pip install -r requirements.txt`

## Run

`python main.py`