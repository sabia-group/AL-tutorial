import shutil
import os, sys
from tqdm import tqdm
import numpy as np
from typing import List, Dict, Union
from ase import Atoms
from ase.io import read, write
from ase.calculators.calculator import Calculator
from contextlib import redirect_stdout, redirect_stderr
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
    with open(os.devnull, 'w') as fnull:
        with redirect_stdout(fnull), redirect_stderr(fnull):
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

def run_qbc(fns_committee:List[str],
            fn_candidates:str,
            fn_train_init:str, # why do we need this?
            n_iter:int,
            config:str,
            ofolder:str="qbc-work", 
            n_add_iter:int=10,
            recalculate_selected:bool=False,
            calculator:Calculator=None):
    """
    Main QbC loop.
    Parameters
    ----------
    fns_committee : List[str]
        List of MACE model files.
    fn_candidates : str
        Filename of the candidates file.
    fn_train_init : str
        Filename of the initial training set.
    n_iter : int
        Number of QbC iterations.
    config: str
        Folder where the MACE configuration files are stored.
        The code will expect the files to be named config.0.yml, config.1.yml, etc.
    ofolder: str
        Folder where to store the QbC results.
    n_add_iter : int
        Number of structures to add to the training set in each iteration.
    recalculate_selected : bool
        If True, the selected structures will be recalculated using the ASE calculator.
    calculator : ASE calculator
        ASE calculator to use for the recalculation of the selected structures.
    """
    # TODO: Add the possibility of attaching a ASE calculator for later when we need to address unlabeled data.
    # TODO: think about striding the candidates to make it more efficient
    # TODO: start from training set size 0?

    print(f'Starting QbC.')
    print(f'{n_iter:d} iterations will be done in total and {n_add_iter:d} will be added every iteration.')

    #os.makedirs('QbC', exist_ok=True)

    candidates:List[Atoms] = read(fn_candidates, index=':')
    training_set = []
    progress_disagreement = []
    for iter in tqdm(range(n_iter)):

        # predict disagreement on all candidates
        print(f'Predicting committee disagreement across the candidate pool.')
        energies = []
        for n, model in enumerate(fns_committee):
            fn_dump = f'{ofolder}/eval_train_{n:02d}.extxyz'
            eval_mace(model, fn_candidates, fn_dump) # Explicit arguments!
            e = extxyz2energy(fn_dump)
            energies.append(e)
        energies = np.array(energies)
        disagreement = energies.std(axis=0)
        avg_disagreement_pool = disagreement.mean()

        # pick the `n_add_iter` highest-disagreement structures
        print(f'Picking {n_add_iter:d} new highest-disagreement data points.')
        idcs_selected = np.argsort(disagreement)[-n_add_iter:]
        print(idcs_selected)
        avg_disagreement_selected = (disagreement[idcs_selected]).mean()
        progress_disagreement.append(np.array([avg_disagreement_selected, avg_disagreement_pool]))
        # TODO: an ASE calculator will come here
        if recalculate_selected:
            assert calculator is not None, 'If a first-principles recalculation of training data is requested, a corresponding ASE calculator must be assigned.'
            print(f'Recalculating ab initio energies and forces for new data points.')
            for structure in candidates[idcs_selected]:
                structure.calc = calculator
                structure.get_potential_energy()
                structure.get_forces()
        #training_set.append([candidates[i] for i in idcs_selected])
        #candidates = np.delete(candidates, idcs_selected)
        # TODO: super ugly, make it better
        for i in idcs_selected:
            training_set.append(candidates[i])
        for i in idcs_selected:
            del candidates[i]

        # dump files with structures
        new_training_set = f'{ofolder}/train-iter.n={iter}.extxyz'
        new_candidates = f'{ofolder}/candidates.n={iter}.extxyz'
        write(new_training_set, training_set, format='extxyz')
        write(new_candidates, candidates, format='extxyz')

        # retrain the committee with the enriched training set
        print(f'Retraining committee.')
        # TODO: add multiprocessing
        # TODO: add model refinement
        for n in range(len(fns_committee)):
            train_mace(f"{config}/config.{n}.yml")

        # update the candidate file name
        fn_candidates = new_candidates

        print(f'Status at the end of this QbC iteration: Disagreement (pool) [eV]    Disagreement (selected) [eV]')
        print(f'                                         {avg_disagreement_pool:06f} {avg_disagreement_selected:06f}')

    # dump final training set
    write(f'{ofolder}/train-final.extxyz', training_set, format='extxyz')
    np.savetxt(f'{ofolder}/disagreement.txt', progress_disagreement)