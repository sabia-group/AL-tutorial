# AL-tutorial
Everything connected to the active learning tutorial for the CNPEM-MPG meeting 07/2025.

## Overview
We are gonna provide a tutorial for ...

## Resources
- Webpage of our research group: https://www.mpsd.mpg.de/research/groups/sabia
- GitHub repository of this tutorial: https://github.com/sabia-group/AL-tutorial
- MACE repository: https://github.com/ACEsuit/mace
- ase repository: https://gitlab.com/ase/ase


## Installation

Create a `conda` environment (not needed if running on a virtual machine):
```bash
conda env create -f environment.yml
```

Activate the `alt` environment (Active Learning Tutorial):
```bash
conda activate alt
pip install -e .
```

Check that the main packages that we need are installed:
```bash
python tests/check.py
```