# Search for Minimal Filling PNNS (GPU Based Backend + New Algorithm)

**Warning:** Some of the code is not well explained and currently some of it needs a human pass by me. So far, it seems the code runs as intended and there are no bugs. BUT, I personally would like to correct some details and ensure the PyTorch code is cleaner and more readable.

**Warning 2:** The README text below may be outdated since not all of it has been updated since.

## Requirements

Required packages are: itertools, numpy, os, ast, math, random, pandas, tqdm, typing, cupy.

Requires CUDA compatible GPU.

## Description

This branch of the repository provides a GPU based improvement to searching for minimal filling architectures. One of the main improvements is a GPU-based approach to computing the dimension of the neurovariety.

The results of all searches can be found in the data folder.

Counterexamples to the unimodal minimal filling architecture conjecture can be found in data/processed/counterexamples.csv


There are two notebooks depending on how much control over the search space one wants: [frontier_search.ipynb](notebooks/frontier_search.ipynb) allows only for searching over $[m,n]^{L-1}$ while [flexible_search.ipynb](notebooks/flexible_search.ipynb) allows on to search over variable widths e.g. $\prod_{i=1}^{L-1}[m_i,n_i]$. Both save the results of the search to csv files such as [2_1_architectures.csv](data/raw/2_1_architectures.csv).

The notebook [analyze_results.ipynb](notebooks/analyze_results.ipynb) stored some analysis of the results we were doing.

Running the notebook [process_csv_files.ipynb](notebooks/process_csv_files.ipynb) produces the .csv files such as [2_1_cleaned_architectures.csv](data/processed/2_1_cleaned_architectures.csv) which display the results of our searches and useful information such as the defect. The notebook also produces [counterexamples.csv](data/processed/counterexamples.csv) which contains all of the counterexamples to the minimal unimodal we could find.

## Disclaimer 

We acknowledge the use of GitHub Copilot (via Visual Studio Code) and Gemini 3.1 in code generation, formatting, and debugging during the writing of this repository. Our process dictated that all AI-assisted code must be rewritten for readability, tested for bugs, and verified line-by-line for validity by humans who take full responsibility. 