# Evolutionary Algorithms - Circles in a Square (CiaS)

This repository contains an implementation of an Evolution Strategy algorithm to solve the Circles in a Square (CiaS) problem. The goal is to find optimal packings of n circles in a unit square by maximizing the minimum distance between any pair of points.

## Team Members

- Vasil Dakov
- Alperen Guncan
- Stiliyan Nanovski
- Todor Slavov
- Ivo Yordanov

## Current Implementation

The current implementation uses an Evolution Strategy (ES) algorithm with:

- Multiple strategy variants (single variance, multiple variance, full variance)
- Basic boundary handling
- Random initialization
- Early stopping

## Possible Improvements

1. Constraint Handling:

   - [ ] Implement boundary repair
   - [ ] Implement constraint domination

2. Initialization:

   - [ ] Develop problem-specific initialization scheme

3. Parameter Optimization:
   - [ ] Optimize population size
   - [ ] Optimize number of generations
   - [ ] Optimize mutation parameters

## Setup and Usage

### Installation

#### Option 1: Using venv (Python's built-in virtual environment)

```bash
# Create and activate virtual environment
python -m venv venv

# On Windows:
venv\Scripts\activate && pip install -r requirements.dev.txt
# On Unix or MacOS:
source venv/bin/activate && pip install -r requirements.dev.txt
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
