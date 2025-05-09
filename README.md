# AL-tutorial
Everything connected to the active learning tutorial for the CNPEM-MPG meeting 07/2025.

## Installation

Create a `conda` environment (not needed if running on a virtual machine):
```bash
conda env create -f environment.yml
```

Activate the `alt` environment (Active Learning Tutorial):
```bash
conda activate alt
```

Check that the main packages that we need are installed:
```bash
python tests/check.py
```