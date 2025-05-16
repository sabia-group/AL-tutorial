
import sys
import numpy as np
from ase.io import read 
from mace.cli.run_train import main as mace_run_train_main          # train a MACE model
from mace.cli.eval_configs import main as mace_eval_configs_main    # evaluate a MACE model

# definition of some helper functions
def extxyz2energy(file:str,keyword:str="MACE_energy"):
    """Extracts the energy values from an extxyz file and returns a numpy array
    """
    atoms = read(file, index=':')
    data = np.zeros(len(atoms),dtype=float)
    for n,atom in enumerate(atoms):
        data[n] = atom.info[keyword]
    return data

def train_mace(config:str):
    """Train a MACE model using the provided configuration file.
    """
    sys.argv = ["program", "--config", config]
    mace_run_train_main()
    
def eval_mace(model:str,infile:str,outfile:str):
    """Evaluate a MACE model.
    """
    sys.argv = ["program", "--config", infile,"--output",outfile,"--model",model]
    mace_eval_configs_main()

def retrain_mace(config:str):
    """Train a MACE model using the provided configuration file.
    """
    sys.argv = ["program", "--config", config]
    mace_run_train_main()