# Evolutionary Algorithms - Circles in a Square (CiaS)

This repository contains an implementation of an Evolution Strategy algorithm to solve the Circles in a Square (CiaS) problem. The goal is to find optimal packings of n circles in a unit square by maximizing the minimum distance between any pair of points.

## Team Members

- Vasil Dakov
- Alperen Guncan
- Stiliyan Nanovski
- Todor Slavov
- Ivo Yordanov

## Baseline

We use the Evolution Strategy (ES) algorithm with:

- Multiple strategy variants (single variance, multiple variance, full variance)
- Random/resampling boundary repair
- Random (normal) initialization
- Early stopping

## Implemented Features

- Boundary Repair
- Constraint Domination
- Initialization Schemes
  - Random
  - Grid
  - Concentric
  - Edge
  - Spiral
- Elitism
- Recombination Strategies
  - Weighted
  - Intermediate
  - Correlated Mutations
- CMA

# Experiments

Our experiments can be found in the `experiments` subfolder. `experiments.py` contains a method that compares all options of a specific feature against each other, with all other options being turned off. This way we can compare the effectiveness of the feature against the baseline model. The calls to those experiments can be found in the `*_comparison.py` files.

We also conducted a grid search to find the best combination of parameters, found in `hyperparameter_optimiser.py`. We limit the search to only using CMA and features that work with CMA, due to the very significant improvements that CMA has over the other variance strategies, such as full covariance. This script can take very long to complete.

## Setup and Usage

### Installation

#### Option 1: Using venv (Python's built-in virtual environment)

```bash
# Create and activate virtual environment
python -m venv venv

# On Windows:
venv\Scripts\activate && pip install -r requirements.txt
# On Unix or MacOS:
source venv/bin/activate && pip install -r requirements.txt
```

#### Option 2: Using Conda

```bash
# Create conda environment and install dependencies in one go
conda env create -f environment.yml
# Activate the environment
conda activate cias
```

### Running the Algorithm

```bash
python main.py
```
