import shutil
import os, sys, glob
import multiprocessing
from datetime import datetime
import time
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
    with open(os.devnull, 'w') as fnull:
        with redirect_stdout(fnull), redirect_stderr(fnull):
            mace_eval_configs_main()

def retrain_mace(config:str):
    """Train a MACE model using the provided configuration file.
    """
    sys.argv = ["program", "--config", config]
    with open(os.devnull, 'w') as fnull:
        with redirect_stdout(fnull), redirect_stderr(fnull):
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

GLOBAL_CONFIG_PATH = None
def train_single_model(n_config):
    train_mace(f"{GLOBAL_CONFIG_PATH}/config.{n_config}.yml")

def run_qbc(init_train_folder:str,
            fn_candidates:str,
            n_iter:int,
            config:str,
            test_dataset:str=None,
            ofolder:str="qbc-work", 
            n_add_iter:int=10,
            recalculate_selected:bool=False,
            calculator:Calculator=None,
            parallel:bool=True):
    """
    Main QbC loop.
    Parameters
    ----------
    fns_committee : List[str]
        List of MACE model files.
    fn_candidates : str
        Filename of the candidates file.
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

    
    #-------------------------#
    # create folders
    #-------------------------#
    folders = [ofolder,f"{ofolder}/eval",f"{ofolder}/structures",f"{ofolder}/models",f"{ofolder}/checkpoints"]
    for f in folders:
        os.makedirs(f, exist_ok=True)
        
    #-------------------------#
    # Copy models and checkpoints to new folder
    #-------------------------#
    copy_files_in_folder(f"{init_train_folder}/checkpoints/",f"{ofolder}/checkpoints/")
    copy_files_in_folder(f"{init_train_folder}/models/",f"{ofolder}/models/")
    
    
    #-------------------------#
    # Banner
    #-------------------------#
    model_dir = os.path.join(ofolder, "models")
    n_committee = len(glob.glob(os.path.join(model_dir, "mace.com=*.model")))
    assert n_committee > 1, "error"

    print(f'Starting QbC.')
    print(f"Number of models: {n_committee:d}")
    print(f"Number of iterations: {n_iter:d}")
    print(f"Number of new candidates at each iteration: {n_add_iter:d}")
    print(f"Candidates file: {fn_candidates}")
    print(f"Test file: {test_dataset}")
    
    #-------------------------#
    # Preparation
    #-------------------------#
    shutil.copy(fn_candidates, f'{ofolder}/candidates.start.extxyz')
    fn_candidates = f'{ofolder}/candidates.start.extxyz'
    
    candidates:List[Atoms] = read(fn_candidates, index=':')
    training_set = []
    progress_disagreement = []
    
    #-------------------------#
    # Look for models
    #-------------------------#
    fns_committee = [None]*n_committee
    for n in range(n_committee):
        fns_committee[n] = f'{ofolder}/models/mace.com={n}.model'
    
    #-------------------------#
    # QbC loop
    #-------------------------#    
    for iter in range(n_iter):
        start_time = time.time()
        start_datetime = datetime.now()
        print(f'\n\t--------------------------------------------------------------------')
        print(f'\tStart of QbC iteration {iter+1:d}/{n_iter:d}\n')
        print(f'\tStarted at: {start_datetime.strftime("%Y-%m-%d %H:%M:%S")}')
    
        # predict disagreement on all candidates
        print(f'\tPredicting committee disagreement across the candidate pool.')
        energies = [None]*len(fns_committee)
        for n, model in enumerate(fns_committee):
            fn_dump = f"{ofolder}/eval/train.model={n}.iter={iter}.extxyz"
            eval_mace(model, fn_candidates, fn_dump) # Explicit arguments!
            e = extxyz2energy(fn_dump)
            energies[n] = e
            
            if test_dataset is not None:
                eval_mace(model, test_dataset, f"{ofolder}/eval/test.model={n}.iter={iter}.extxyz")
            
        energies = np.array(energies)
        disagreement = np.std(energies,axis=0)
        avg_disagreement_pool = np.mean(disagreement) # orange

        # pick the `n_add_iter` highest-disagreement structures
        print(f'\tPicking {n_add_iter:d} new highest-disagreement data points.')
        idcs_selected = np.argsort(disagreement)[-n_add_iter:]
        # print("\t",idcs_selected)
        
        disagreement_selected = disagreement[idcs_selected]
        avg_disagreement_selected = np.mean(disagreement_selected)
        
        progress_disagreement.append(np.array([ avg_disagreement_selected,\
                                                avg_disagreement_pool,\
                                                np.std(disagreement_selected),\
                                                np.std(disagreement),\
                                                len(training_set),len(candidates)]))
        
        #-------------------------#
        # Recalculate energies and forces for selected structures
        #-------------------------#
        # TODO: an ASE calculator will come here
        if recalculate_selected:
            assert calculator is not None, 'If a first-principles recalculation of training data is requested, a corresponding ASE calculator must be assigned.'
            print(f'\tRecalculating ab initio energies and forces for new data points.')
            for structure in candidates[idcs_selected]:
                structure.calc = calculator
                structure.get_potential_energy()
                structure.get_forces()
        
        
        training_set.extend([candidates[i] for i in idcs_selected])
        candidates = [item for i, item in enumerate(candidates) if i not in idcs_selected]

        # dump files with structures
        new_training_set = f'{ofolder}/structures/train-iter.n={iter}.extxyz'
        new_candidates = f'{ofolder}/structures/candidates.n={iter}.extxyz'
        write(new_training_set, training_set, format='extxyz')
        write(new_candidates, candidates, format='extxyz')
        
        # update the training set file name
        shutil.copy(new_training_set, f'{ofolder}/train-iter.extxyz') # MACE will use this file to train
        # update the candidate file name
        fn_candidates = new_candidates

        #-------------------------#
        # Training
        #-------------------------#
        # retrain the committee with the enriched training set
        print(f'\tRetraining committee.')
        # TODO: add model refinement: check that it is actually done
        global GLOBAL_CONFIG_PATH
        GLOBAL_CONFIG_PATH = config
        
        if parallel: # parallel version: it should take around 25s 
            with multiprocessing.Pool(processes=multiprocessing.cpu_count()) as pool:
                pool.map(train_single_model, range(n_committee))
                
        else: # serial version: it should take around 1m
            for n in range(n_committee):
                train_single_model(n)
                
        # clean_output(ofolder,n_committee)

        print(f'\n\tResults of QbC iteration {iter+1:d}/{n_iter:d}:')
        print(f'\t               Disagreement (pool): {avg_disagreement_pool:06f} eV')
        print(f'\t           Disagreement (selected): {avg_disagreement_selected:06f} eV')
        print(f'\t                New training set size: {len(training_set):d}')
        print(f'\t               New candidate set size: {len(candidates):d}')
        
        
        end_time = time.time()
        end_datetime = datetime.now()
        elapsed = end_time - start_time

        print(f'\n\tEnd of QbC iteration {iter+1:d}/{n_iter:d}')
        print(f'\tEnded at:   {end_datetime.strftime("%Y-%m-%d %H:%M:%S")}')
        print(f'\tDuration:   {elapsed:.2f} seconds')
        
        header = "\
selected-mean\n\
pool-mean\n\
selected-std\n\
pool-std\n\
training-set-size\n\
candidate-set-size\
"
        np.savetxt(f'{ofolder}/disagreement.txt', progress_disagreement,header=header,fmt='%12.8f')
        
    #-------------------------#
    # Finalize
    #-------------------------#
    print(f'\n\t--------------------------------------------------------------------')
    print(f'\tEnd of QbC loop.\n')
    print(f'\tFinal training set size: {len(training_set):d}')
    print(f'\tFinal candidate set size: {len(candidates):d}')
            
    # dump final training set
    write(f'{ofolder}/train-final.extxyz', training_set, format='extxyz')
    # np.savetxt(f'{ofolder}/disagreement.txt', progress_disagreement)
    
    os.remove(f'{ofolder}/train-iter.extxyz')
    
def clean_output(folder,n_committee):
    # remove useless files
    for filename in os.listdir(f'{folder}/log'):
        if filename.endswith('_debug.log'):
            file_path = os.path.join(f'{folder}/log', filename)
            os.remove(file_path)
            
    for n in range(n_committee):
        
        # models
        filenames = [f"{folder}/models/mace.com={n}.model",
                    f"{folder}/models/mace.com={n}_compiled.model",
                    f"{folder}/models/mace.com={n}_stagetwo.model"]
        for filename in filenames:
            if os.path.exists(filename):
                os.remove(filename)
        
        if os.path.exists(f"{folder}/models/mace.com={n}_stagetwo_compiled.model"):
            os.rename(f"{folder}/models/mace.com={n}_stagetwo_compiled.model",f"{folder}/models/mace.n={n}.model")
        
    for filename in os.listdir(f'{folder}/results'):
        if filename.endswith('.txt') or filename.endswith('stage_one.png'):
            file_path = os.path.join(f'{folder}/results', filename)
            os.remove(file_path)
            
def copy_files_in_folder(src,dst):
    [shutil.copy(f"{src}/{f}", dst) for f in os.listdir(src) if os.path.isfile(f"{src}/{f}")]