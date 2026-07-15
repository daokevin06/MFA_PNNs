# GPU Search for Minimal Filling Architectures of Polynomial Neural Networks

This repository contains computational tools for studying **filling**, **defective**, and **minimal filling architectures** of fully connected polynomial neural networks with monomial activation.

The main computational backend evaluates the dimension of a polynomial-neural-network neurovariety using exact finite-field arithmetic on an NVIDIA GPU. The search notebooks use this backend to enumerate candidate architectures, identify filling architectures, and search for minimal filling architectures across prescribed regions of architecture space.

**Requirements:** This current branch requires a GPU. Unfortunately, due to its better speed, the data on this branch is more complete than the main branch. I do not plan to update the main branch much more due to this, but this *does* mean one should use this branch for the current results.

## Status

*This is actively being modified code.* The computations are intended to provide:

* large-scale experimental data;

* candidate minimal filling architectures;

* candidate defective architectures;

* tests of conjectures concerning the shape of minimal filling architectures;

* independently reproducible dimension calculations.

A row marked `is_minimal=True` in a search CSV may initially mean only that the architecture is minimal among the filling architectures encountered in that search. Mathematically rigorous verifications for minimality requires checking all relevant maximal subarchitectures are nonfilling. The repository contains additional tools for completing these predecessor computations. Search results should therefore be interpreted as computational evidence unless the required predecessor checks have been completed.

**To-do:** I plan to add columns for identifiability results, columns for activation threshold, and other geometric properties. If you know a property that could be reasonably added and seems worthwhile to add, let me know! Not all properties are interesting e.g. ``is it a singular variety?" is not interesting because neurovarieties that are nonfilling are always singular at the origin or if nonfilling, will often contain a subarchitecture as part of its singular locus.

## Mathematical setting

Consider an architecture $\mathbf d=(d_0,d_1,\ldots,d_L)$ with coordinatewise hidden-layer activation $\sigma:z\longmapsto z^r$ and a linear final layer. The represented polynomial map has degree $D=r^{L-1}$. Its output belongs to an ambient space (of coefficients of the homogeneous polynomials) of dimension $d_L\binom{D+d_0-1}{d_0-1}$. The Zariski closure of the set of polynomial maps represented by the architecture is denoted by $\mathcal V_{\mathbf d,r}$. It is called the *neurovariety*. The architecture is called **filling** when $\dim \mathcal V_{\mathbf d,r} = d_L\binom{D+d_0-1}{d_0-1}$. A filling architecture is **minimal filling** if decreasing any hidden width produces a nonfilling architecture. Architectures are ordered coordinatewise in their hidden widths.

Currently, I am focused on isolating the following architectures

- minimal filling architectures

- minimal filling architectures which are *nonunimodal*

- architectures with codimension $1$ neuroviarieties


## Main computational method

The dimension backend is implemented in [`notebooks/dim_backprop_gpu_only.py`](notebooks/dim_backprop_gpu_only.py).

At randomly selected weights over a finite field, the dimension is computed as the rank of the differential of the network parameterization. The implementation accelerates this calculation in several ways:

1. **Unisolvent evaluation points.**
   Homogeneous degree $D$ polynomials are evaluated on an explicit unisolvent set of exactly $\binom{D+d_0-1}{d_0-1}$ points. Here, unisolvent to me means the same thing as in https://en.wikipedia.org/wiki/Unisolvent_point_set.

2. **Batched differentiation.**
   All input samples and all output-coordinate pullbacks are differentiated simultaneously using CuPy tensors.

3. **No coefficient reconstruction.**
   The rank of the evaluated Jacobian equals the rank of the coefficient Jacobian because evaluation on the chosen point set is invertible. The code therefore avoids monomial enumeration and pseudoinverse-based interpolation.

4. **Exact finite-field rank.**
   The Jacobian rank is computed by modular Gaussian elimination on the GPU.

5. **Multiple-prime checking.**
   The calculation is repeated over multiple primes. A disagreement between the resulting ranks raises an error. **Choosing larger primes may lead to slow-downs. It is always advised to pick more than one prime to ensure higher likelihood of correctness of results.**

A detailed explanation is available in [`docs/gpu_improvements.md`](docs/gpu_improvements.md).

## Requirements

The main GPU workflow requires:

* Python 3.10 or later;
* an NVIDIA GPU;
* a working NVIDIA driver;
* CuPy compiled for the installed CUDA version;
* Jupyter;
* pandas;
* tqdm.

The main GPU search path does **not** require SageMath, Magma, or a C++ compiler.

One older verification notebook still contains SageMath-based code and is discussed separately below. **(To be done: This will be updated later.)**

## Installation

Clone the GPU branch:

```bash
git clone --branch GPU https://github.com/daokevin06/MFA_PNNs.git
cd MFA_PNNs
```

Create a virtual environment:

```bash
python -m venv .venv
```

Activate it on Windows:

```powershell
.venv\Scripts\activate
```

Activate it on Linux or macOS:

```bash
source .venv/bin/activate
```

Install the common dependencies:

```bash
python -m pip install --upgrade pip
python -m pip install jupyterlab pandas tqdm
```

Install the CuPy package matching the CUDA installation. For example, for a CUDA 12.x environment:

```bash
python -m pip install cupy-cuda12x
```

The appropriate CuPy package may differ for other CUDA versions.

## Testing the GPU backend

Start Python or a notebook from the `notebooks` directory:

```bash
cd notebooks
```

Then run:

```python
from dim_backprop_gpu_only import compute_dimension, gpu_information

print(gpu_information())

result = compute_dimension(
    network_widths=[2, 3, 1],
    network_exponent=2,
    verbose=True,
)

print(result)
```

The returned tuple has the form

```text
(
    architecture,
    activation_exponent,
    ambient_dimension,
    expected_dimension,
    computed_dimension,
    defect,
)
```

For example, a defect of zero means that the computed dimension agrees with the symmetry-corrected expected dimension. Filling is determined by comparing the computed dimension with the ambient dimension.

## Recommended workflows

### 1. Optimized search with theoretical pruning

Use [`notebooks/faster_search.ipynb`](notebooks/faster_search.ipynb) for the most structured first search. It is unproven, but it seems likely that all minimal filling architectures sit below the threshold in KTB's paper.

This notebook uses:

* ambient-dimension bounds;
* parameter-count bounds;
* recursive Kileel–Trager–Bruna bounds;
* coordinatewise necessary lower bounds;
* proven finite upper bounds for hidden widths;
* selected exact shallow quadratic bounds;
* persistent CSV output.

This is the recommended starting point for systematic searches over architecture space.

### 2. Flexible search over custom boxes

Use [`notebooks/flexible_search_replace.ipynb`](notebooks/flexible_search_replace.ipynb) when explicit control over every hidden-layer range is desired.

It supports regions of the form $\prod_{i=1}^{L-1}[m_i,n_i],$ so different hidden layers may have different lower and upper bounds. Currenty, I've got the KTB bounds as the default.

The notebook also supports:

* global width bounds;
* custom per-layer bounds;
* resuming from an existing CSV;
* skipping previously computed architectures;
* periodic saving;
* safe interruption;
* updating minimality flags relative to the computed search region.

### 3. Complete predecessor data in existing CSV files

Use [`notebooks/MFA_Sequential_CSV_Dimension_Fill_GPU_Only_Windows_Safe.ipynb`](notebooks/MFA_Sequential_CSV_Dimension_Fill_GPU_Only_Windows_Safe.ipynb) to complete missing dimension computations in existing files under `data/raw/`. For each recorded minimal filling architecture, the notebook enumerates architectures in the relevant coordinatewise predecessor box and computes missing dimensions sequentially.

The notebook includes:

* in-place CSV updates;
* checkpointing after a configurable number of new rows;
* skipping rows with valid existing results;
* safe handling of `KeyboardInterrupt`;
* retry logic for temporary Windows file locks;
* optional stopping when a filling strict predecessor contradicts a recorded minimality flag.

This workflow is useful for upgrading older search data without rerunning the original search.

### 4. Process raw search data

Use [`notebooks/process_csv_files.ipynb`](notebooks/process_csv_files.ipynb) to transform raw search files into cleaned result files.

The notebook computes or updates quantities such as:

* expected dimension;
* defect;
* filling status;
* minimality flags;
* unimodality.

Processed files are written to [`data/processed/`](data/processed/). The file [`data/processed/counterexamples.csv`](data/processed/counterexamples.csv) collects architectures identified by the processing workflow as nonunimodal minimal filling candidates. These should be checked against the available predecessor data before being treated as rigorously verified counterexamples.

### 5. Explore the results

Use [`notebooks/analyze_results.ipynb`](notebooks/analyze_results.ipynb) as a sandbox for examining the CSV data, filtering by depth and exponent, and studying nonunimodal architectures.

## Repository structure

```text
MFA_PNNs/
├── README.md
├── data/
│   ├── raw/
│   │   └── *_architectures.csv
│   └── processed/
│       ├── *_cleaned_architectures.csv
│       └── counterexamples.csv
├── docs/
│   └── gpu_improvements.md
└── notebooks/
    ├── dim_backprop_gpu_only.py
    ├── faster_search.ipynb
    ├── flexible_search_replace.ipynb
    ├── frontier_search_gpu.ipynb
    ├── MFA_Sequential_CSV_Dimension_Fill_GPU_Only_Windows_Safe.ipynb
    ├── process_csv_files.ipynb
    ├── analyze_results.ipynb
    ├── verify_minimal.ipynb
    └── is_unimodal.py
```

## Notebook summary

| File                                                            | Purpose                                                     | Status                       |
| --------------------------------------------------------------- | ----------------------------------------------------------- | ---------------------------- |
| `dim_backprop_gpu_only.py`                                      | Exact finite-field GPU dimension computation                | Core backend                 |
| `faster_search.ipynb`                                           | Search with theoretical lower and upper bounds              | Recommended                  |
| `flexible_search_replace.ipynb`                                 | Search with custom per-layer boxes and resumable CSV output | Recommended                  |
| `MFA_Sequential_CSV_Dimension_Fill_GPU_Only_Windows_Safe.ipynb` | Fill missing predecessor dimensions in existing CSV files   | Recommended for verification |
| `process_csv_files.ipynb`                                       | Clean raw results and generate derived datasets             | Postprocessing               |
| `analyze_results.ipynb`                                         | Exploratory analysis of search results                      | Analysis                     |
| `frontier_search_gpu.ipynb`                                     | Earlier parameter-frontier search implementation            | Legacy/experimental          |
| `verify_minimal.ipynb`                                          | Earlier SageMath-based predecessor verification             | Legacy; needs modernization  |
| `is_unimodal.py`                                                | Test whether an architecture is unimodal                    | Utility                      |

## CSV format

Raw search files generally contain columns of the following form:

| Column               | Meaning                                                     |
| -------------------- | ----------------------------------------------------------- |
| `h`                  | Network depth used by the search                            |
| `exponent`           | Hidden-layer activation degree (r)                          |
| `architecture`       | Full architecture ([d_0,\ldots,d_L])                        |
| `num_parameters`     | Number of scalar weight parameters                          |
| `dimension_computed` | Computed dimension of the neurovariety                      |
| `ambient_dimension`  | Dimension of the ambient polynomial space                   |
| `is_full_dimension`  | Whether the computed dimension equals the ambient dimension |
| `is_minimal`         | Whether the architecture is currently marked minimal        |
| `is_unimodal`        | Whether the architecture is unimodal, when available        |

Processed files may contain additional derived columns, including expected dimension and defect.

## Reproducibility and interpretation

The modular row reduction is exact over each selected finite field. However, the generic dimension is inferred by evaluating the differential at randomly generated weights. Consequently:

* an unlucky weight choice may produce a rank below the generic rank;
* exceptional behavior may occur for a particular finite field;
* agreement over several primes provides a useful consistency check but is not by itself a formal proof over characteristic zero;
* important examples should be rerun with additional seeds or primes;
* claimed minimality should be checked by evaluating the necessary strict predecessors.

The random seed and finite-field primes can be changed through the arguments of `compute_dimension`.

## Memory considerations

For architecture (\mathbf d), the evaluated Jacobian has approximately $d_L\binom{r^{L-1}+d_0-1}{d_0-1}$ rows and $\sum_{i=1}^{L}d_{i-1}d_i$ columns.

Deep architectures, large input dimension, or large activation degree can therefore require substantial GPU memory. For example, $(3,80,80,80,80,3)$ and $r=6$ runs into memory issues. The rank routine supports chunked row elimination through the `rank_workspace_bytes` argument, but the Jacobian itself must still fit in available device memory.

## Related work

The implementation builds on dimension-computation methods and software developed for the study of polynomial neural networks, including the material accompanying: [Polynomial Neural Networks — MathRepo](https://mathrepo.mis.mpg.de/PolynomialNeuralNetworks/). The principal computational changes in this repository are the use of an explicit unisolvent evaluation set, batched GPU differentiation, direct rank computation on the evaluated Jacobian, and GPU-parallel modular elimination.

<!-- ## Contributing

Bug reports, independently reproduced calculations, improved theoretical pruning bounds, and alternative exact rank implementations are welcome. When reporting a computational result, please include:

* the architecture;
* activation exponent;
* finite-field primes;
* random seed;
* GPU model;
* CuPy and CUDA versions;
* computed rank and ambient dimension;
* whether all relevant predecessors were checked. -->

## AI-assisted development disclosure

GitHub Copilot and ChatGPT were used during portions of code generation, formatting, and debugging. AI-generated suggestions are not treated as authoritative. **The maintainer takes responsibility for reviewing, testing, and validating the code and mathematical computations**. Independent verification is encouraged.