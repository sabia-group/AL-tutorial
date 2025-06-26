# Machine learning interatomic potentials – active learning techniques
Everything connected to the active learning tutorial for the [CNPEM-MPG](https://pages.cnpem.br/ilum-maxplanck-meeting/) meeting 07/2025.

## Overview
This tutorial illustrates the basic concepts of committee-based active learning using MACE potentials to represent the potential energy surfaces in simple, illustrative systems. 

You will find two tutorials in the `notebook` folder:
- `1-zundel.ipynb`: apply Query by Committee to the zundel cation from labeled data
- `2-eigen.ipynb`: apply Query by Committee to the eigen cation from unlabeled data

## Resources
- Website of the Workshop: https://pages.cnpem.br/ilum-maxplanck-meeting/
- Webpage of our research group (SAbIA): https://www.mpsd.mpg.de/research/groups/sabia
- GitHub repository of this tutorial: https://github.com/sabia-group/AL-tutorial
- MACE repository: https://github.com/ACEsuit/mace
- `ase` repository: https://gitlab.com/ase/ase


## Installation

### System Dependencies

This tutorial requires the system package `python3-tk` for GUI support (used by `ase` and related tools).  
Please install it before proceeding:

```bash
sudo apt-get install python3-tk
```

### Option 1: Using `pip` (recommended for simplicity)

Create and activate a virtual environment and install the package and its dependencies through the `pyproject.toml` file:
```bash
mkdir -p ~/venv
python -m venv ~/venv/alt
source ~/venv/alt/bin/activate  
# On Windows use: %USERPROFILE%\venv\alt\Scripts\activate
python -m pip install -e .
```

### Option 2: Using `conda` (alternative approach)
Create a `conda` environment using the provided `environment.yaml` file and activate the `alt` environment:
```bash
conda env create -f environment.yml
conda activate alt
```


### Verify Installation
Check that the main packages that we need are installed:
```bash
python tests/check.py
```

## References
These tutorials are based on the following work:
- [Schran C., Brezina K., Marsalek O. JCP, 153, 104105, 2020](https://doi.org/10.1063/5.0016004)
