import shutil
import os, sys
import numpy as np
from typing import List, Dict, Union
from ase import Atoms
from ase.io import read, write
from mace.cli.run_train import main as mace_run_train_main          # train a MACE model
from mace.cli.eval_configs import main as mace_eval_configs_main    # evaluate a MACE model

__all__ = ["extxyz2energy", "train_mace", "eval_mace", "run_aims"]

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
    
def run_aims(structures:List[Atoms],folder:str,command:str,control:str)->List[Atoms]:
    """
    Run AIMS on a list of structures.
    Parameters
    ----------
    structures : List[Atoms]
        List of ASE Atoms objects.
    folder : str
        Folder where the AIMS input files are stored.
    command: str
        'mpirun -n 4 aims.x'
    control: str
        filepath of a control.in file (necessary for FHI-aims to run).
    """
    # create folder where to work and store input and output files
    os.makedirs(folder, exist_ok=True)
    output = [None]*len(structures)
    for n,structure in enumerate(structures):
        # create folder
        nfolder = f"{folder}/structure-n={n}"
        os.makedirs(nfolder, exist_ok=True)
        # create geometry.in
        ifile = f"{nfolder}/geometry.in"
        write(ifile,structure,format="aims")
        # copy control.in
        shutil.copy(control,f"{nfolder}/control.in")
        # run FHI-aims
        _run_single_aims(nfolder,command)
        # read output file
        ofile = f"{nfolder}/aims.out"
        atoms = read(ofile, format="aims-output")
        # this is necessary to read the forces and energy from the output file correctly    
        output[n] = _correct_read(atoms) 
        
    return output
        
def _run_single_aims(workdir:str,command:str)->Atoms:
    """
    Run AIMS on a single structure.
    Parameters
    ----------
    structure : Atoms
        ASE Atoms object.
    folder : str
        Folder where the AIMS input files are stored.
    aims_path: str
        Path to the AIMS executable.
    """
    original_folder = os.getcwd()
    os.chdir(workdir)
    os.system(f"ulimit -s unlimited && {command} > aims.out")
    os.chdir(original_folder)
    
def _correct_read(atoms:Atoms)->Atoms:
    if atoms.calc is not None:
        results:Dict[str,Union[float,np.ndarray]] = atoms.calc.results
        for key,value in results.items():
            if key in ['energy','free_energy','dipole','stress']:
                atoms.info[key] = value
            elif key in ['forces']:
                atoms.arrays[key] = value
            else: 
                atoms.info[key] = value
    atoms.calc = None 
    return atoms