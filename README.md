# Search for Minimal Filling PNNS

## Package Requirements

Sage is required to run the scripts such as [dim_backprop.py](dim_backprop.py). 

Other required packages are: itertools, numpy, os, ast, math, random, pandas, tqdm.

## Description

A repository containing work involved in searching for a counterexample to the minimal unimodal conjecture.

The results of our search can be found in the data folder.

Counterexamples to the unimodal minimal filling architecture conjecture can be found in data/processed/counterexamples.csv

The main tool for computing dimension is the code in dim_backprop.py taken from work done by [Kaie Kubjas, Jiayi Li, and Maximilian Wiesmann](https://mathrepo.mis.mpg.de/PolynomialNeuralNetworks/).

There are two notebooks depending on how much control over the search space one wants: [frontier_search.ipynb](notebooks/frontier_search.ipynb) allows only for searching over $[m,n]^{L-1}$ while [flexible_search.ipynb](notebooks/flexible_search.ipynb) allows on to search over variable widths e.g. $\prod_{i=1}^{L-1}[m_i,n_i]$. Both save the results of the search to csv files such as [2_1_architectures.csv](data/raw/2_1_architectures.csv).

The code in [is_unimodal.py](is_unimodal.py) simply checks if a string e.g. (2,3,4,5,4,6,4,1) is unimodal or not.

The notebook [analyze_results.ipynb](notebooks/analyze_results.ipynb) stored some analysis of the results we were doing.

The notebooks [2_1_counterexample_verify.ipynb](notebooks//2_1_counterexample_verify.ipynb) and [2_2_counterexample_verify.ipynb](notebooks//2_2_counterexample_verify.ipynb) contain code which verify that $(2,3,4,5,4,6,4,1)$ and $(2,3,4,4,10,17,11,12,4,2)$ are counterexamples to the minimal unimodal conjecture. There is also notebook [r3_counterexample_verify.ipynb](notebooks/r3_counterexample_verify.ipynb) which verifies a counterexample when $r=3$.

The notebook [verify_proposition.ipynb](notebooks/verify_proposition.ipynb) verifies that certain architectures are filling by checking the the Jacobian has maximal rank over $\mathbb{F}_p$ for $p$ a sufficiently large prime.

Running the notebook [process_csv_files.ipynb](notebooks/process_csv_files.ipynb) produces the .csv files such as [2_1_cleaned_architectures.csv](data/processed/2_1_cleaned_architectures.csv) which display the results of our searches and useful information such as the defect. The notebook also produces [counterexamples.csv](data/processed/counterexamples.csv) which contains all of the counterexamples to the minimal unimodal we could find.

## Disclaimer 

We acknowledge the use of GitHub Copilot (via Visual Studio Code) and Gemini 3.1 in code generation, formatting, and debugging during the writing of this repository. Our process dictated that all AI-assisted code must be rewritten for readability, tested for bugs, and verified line-by-line for validity by humans who take full responsibility. 