## Overview

This software repository contains an experimental software implementation of
algorithms accompanying the paper 
"Clustering with fair-center representation: parameterized approximation
algorithms and heuristics". The software is written in the Python programming language.

The source code is subject to MIT license, see LICENSE.txt for copyright details. If you make use of this source-code, cite the following article.

```
Suhas Thejaswi, Ameet Gadekar, Bruno Ordozgoiti, and Michał Osadnik. 2022.
Clustering with fair-center representation: parameterized approximation
algorithms and heuristics. In Proceedings of the 28th ACM SIGKDD Conference on
Knowledge Discovery and Data Mining (KDD ’22), August 14–18, 2022, Washington,
DC, USA. ACM, New York, NY, USA, 11 pages. https://doi.org/10.1145/3534678.3539487
```

## Requirements

See `requirements.txt` for python package requirements.

Before running commands use:

```bash
export PYTHONPATH=$PATHONPATH:`pwd`
```

in the main project directory.

## General comments

The source code is written in a python programming language. The files are names
to ensure the corresponding functionality of the implementation, for instance
file `kmedian_coresets.py` contains implementation for generating coresets for the k-median problem.
Additionally, each file contains test stubs to demonstrate the usage of APIs.

## Experiments

Experiments available in this repository are divided into two groups that have been described in the article.

- Approximation algorithms / heuristics
    - Exhaustive search + Local search (Heuristic)
    - Exhaustive search + FPT algorithm (3-appx, we do not provide submodular reduction)
    - Linear Program + Local Search (Heuristic)
- Bicriteria

To run those experiments on computational node, one might consider using `nohup` program that makes the command not stop with ssh session.
Together with overview of results on the terminal, programs are generating output files that we eventually used for drawing charts.

### Exhaustive search + Local search

```bash
cd es_ls
python scalability.py
```

This command does not support any additional parameters. 

Together with scalability batches, there is available `test_es_ls_complete.py`. This file is example of usage of `es_local_search.py`.

### Exhaustive search + FPT algorithm

```bash
cd es_fpt
python scalability.py
```

This command does not support any additional parameters.

Together with scalability batches, there is available `test_es_fpt.py`. This file is example of usage of `es_fpt_3apx.py`.

### Linear Program + Local Search

```bash
cd lp_ls
python scalability.py
```

This command does not support any additional parameters.

Together with scalability batches, there is available `test_lp_ls_complete.py`. This file is example of usage of `lp_local_search.py`.

### Bicriteria

```bash
cd bicriteria
python scalability.py [--batch_type, --objective, --results-dir]
```

Bicriteria runner is designed to perform a various range of tests.
Together with a complete bicriteria flow (parametrized by the objective function - `kmeans` or `kmedian`,
this is possible to run each element of the algorithm unaccompanied. 
Particularly, the `optimal` part (not restricted by requirements) and `feasibility` might be verified independently.
Parameters below allows to adjust the outcome accordingly:

- `batch-type`: `feasibility`, `optimal`, or `bicriteria` (defaults to `feasibility`)
- `objective`: `kmedian` or `kmeans` (defaults to `kmedian`)
- `results`: the directory to store results (defaults to `exp-results`)

