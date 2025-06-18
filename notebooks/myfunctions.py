import shutil
import os, sys, glob
import multiprocessing
from datetime import datetime
import time
from contextlib import contextmanager
import numpy as np
from typing import List, Dict, Union
from ase import Atoms
from ase.io import read, write
from ase.calculators.calculator import Calculator
from contextlib import redirect_stdout, redirect_stderr
from mace.cli.run_train import main as mace_run_train_main          # train a MACE model
from mace.cli.eval_configs import main as mace_eval_configs_main    # evaluate a MACE model

__all__ = ["extxyz2energy", "extxyz2array", "train_mace", "eval_mace", "run_aims"]

#-------------------------#
# definition of some helper functions
def extxyz2energy(file:str,keyword:str="MACE_energy"):
    """Extracts the energy values from an extxyz file and returns a numpy array
    """
    atoms = read(file, index=':')
    data = np.zeros(len(atoms),dtype=float)
    for n,atom in enumerate(atoms):
        data[n] = atom.info[keyword]
    return data

#-------------------------#
def extxyz2array(file:str,keyword:str="MACE_forces"):
    """Extracts the energy values from an extxyz file and returns a numpy array
    """
    atoms = read(file, index=':')
    data = [None]*len(atoms)
    for n,atom in enumerate(atoms):
        data[n] = atom.arrays[keyword]
    try:
        return np.array(data)
    except:
        return data

#-------------------------#
def train_mace(config:str):
    """Train a MACE model using the provided configuration file.
    """
    sys.argv = ["program", "--config", config]
    with open(os.devnull, 'w') as fnull:
        with redirect_stdout(fnull), redirect_stderr(fnull):
            mace_run_train_main()
    
#-------------------------#
def eval_mace(model:str,infile:str,outfile:str):
    """Evaluate a MACE model.
    """
    sys.argv = ["program", "--config", infile,"--output",outfile,"--model",model]
    with open(os.devnull, 'w') as fnull:
        with redirect_stdout(fnull), redirect_stderr(fnull):
            mace_eval_configs_main()

# #-------------------------#
# def retrain_mace(config:str):
#     """Train a MACE model using the provided configuration file.
#     """
#     sys.argv = ["program", "--config", config]
#     with open(os.devnull, 'w') as fnull:
#         with redirect_stdout(fnull), redirect_stderr(fnull):
#             mace_run_train_main()
   
#-------------------------# 
def run_single_aims_structure(structure: Atoms, folder: str, command: str, control: str) -> Atoms:
    """
    Run FHI-aims on a single structure.
    
    Parameters
    ----------
    structure : Atoms
        ASE Atoms object to run.
    folder : str
        Folder where the calculation will run.
    command : str
        Command to execute FHI-aims, e.g. 'mpirun -n 4 aims.x'.
    control : str
        Path to a control.in file.
        
    Returns
    -------
    Atoms
        Structure with energy/forces info read from the output.
    """
    # Ensure working directory exists
    os.makedirs(folder, exist_ok=True)

    # Prepare input files
    geom_path = os.path.join(folder, "geometry.in")
    ctrl_path = os.path.join(folder, "control.in")
    out_path = os.path.join(folder, "aims.out")

    write(geom_path, structure, format="aims")
    shutil.copy(control, ctrl_path)

    # Run FHI-aims
    _run_single_aims(folder, command)

    # Read result and return corrected Atoms object
    try:
        atoms = read(out_path, format="aims-output")
    except Exception as err:
        raise ValueError(f"An error occcurred while reading '{out_path}'")
    return _correct_read(atoms)

#-------------------------#
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
     
#-------------------------#   
def _run_single_aims(workdir: str, command: str) -> Atoms:
    """
    Run AIMS on a single structure.
    Parameters
    ----------
    workdir : str
        Folder where the AIMS input files are stored.
    command : str
        Full AIMS execution command.
    """
    original_folder = os.getcwd()
    try:
        os.chdir(workdir)
        # Suppress both stdout and stderr
        #os.system(f"ulimit -s unlimited && {command} ")
        os.system(f"ulimit -s hard && {command} ")
    finally:
        os.chdir(original_folder)
    
#-------------------------#
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

#-------------------------#
GLOBAL_CONFIG_PATH = None
def train_single_model(n_config):
    train_mace(f"{GLOBAL_CONFIG_PATH}/config.{n_config}.yml")
    
def ab_initio(args):
    structure, calculator = args
    assert isinstance(structure,Atoms), "wrong data type"
    assert isinstance(calculator,Calculator), "wrong data type"
    structure.calc = calculator
    structure.get_potential_energy()
    structure.get_forces()
    

def run_qbc(init_train_folder:str,
            init_train_file:str,
            fn_candidates:str,
            n_iter:int,
            config:str,
            test_dataset:str=None,
            ofolder:str="qbc-work", 
            n_add_iter:int=10,
            recalculate_selected:bool=False,
            calculator_factory:callable=None,
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
    
    if recalculate_selected:
        assert calculator_factory is not None, "You need to provide as ASE calculator if you want to recalculate energy and forces on the fly."

    #-------------------------#
    # Folders preparation
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
    n_committee = len(glob.glob(os.path.join(model_dir, "mace.com=*_compiled.model")))
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
    training_set:List[Atoms] = read(init_train_file,index=':')
    progress_disagreement = []
    
    #-------------------------#
    # Look for models
    #-------------------------#
    fns_committee = [None]*n_committee
    for n in range(n_committee):
        fns_committee[n] = f'{ofolder}/models/mace.com={n}_compiled.model'
    
    #-------------------------#
    # QbC loop
    #-------------------------#    
    for iter in range(n_iter):
        start_time = time.time()
        start_datetime = datetime.now()
        print(f'\n\t--------------------------------------------------------------------')
        print(f'\tStart of QbC iteration {iter+1:d}/{n_iter:d}\n')
        print(f'\tStarted at: {start_datetime.strftime("%Y-%m-%d %H:%M:%S")}')
        
        #-------------------------#
        # 1) Model evaluation
        #-------------------------# 
    
        # predict disagreement on all candidates
        # KB: working version of using force disagreement instead of energy disagreement
        print(f'\tPredicting committee disagreement across the candidate pool.')
        #energies = [None]*len(fns_committee)
        forces = []
        for n, model in enumerate(fns_committee):
            fn_dump = f"{ofolder}/eval/train.model={n}.iter={iter}.extxyz"
            eval_mace(model, fn_candidates, fn_dump) # Explicit arguments!
            #e = extxyz2energy(fn_dump)
            #energies[n] = e
            f = extxyz2array(fn_dump)
            forces.append(f)
            
            if test_dataset is not None:
                eval_mace(model, test_dataset, f"{ofolder}/eval/test.model={n}.iter={iter}.extxyz")
                
        #-------------------------#
        # 2) Disagreement calculation
        #-------------------------# 
            
        #energies = np.array(energies)
        forces = np.array(forces)
        #disagreement = forces.std(axis=0)
        dforces = forces - forces.mean(axis=0)[None, ...]
        disagreement_atomic = np.sqrt( ( (dforces**2).sum(axis=3) ).mean(axis=0) )
        disagreement = disagreement_atomic.mean(axis=1)
        avg_disagreement_pool = disagreement.mean()

        # pick the `n_add_iter` highest-disagreement structures
        print(f'\tPicking {n_add_iter:d} new highest-disagreement data points.')
        #idcs_selected = np.argsort(disagreement.mean(axis=(1, 2)))[-n_add_iter:]
        idcs_selected = np.argsort(disagreement)[-n_add_iter:]
        # print("\t",idcs_selected)

        
        disagreement_selected = disagreement[idcs_selected]
        avg_disagreement_selected = disagreement_selected.mean()
        
        progress_disagreement.append(np.array([ avg_disagreement_selected,\
                                                avg_disagreement_pool,\
                                                np.std(disagreement_selected),\
                                                np.std(disagreement),\
                                                len(training_set),len(candidates)]))
        
        to_evaluate:List[Atoms] = [candidates[i] for i in idcs_selected]
        
        #-------------------------#
        # 3.a) Energies and forces re-calculation (optional)
        #-------------------------#
        if recalculate_selected:
            assert calculator_factory is not None, \
                'If a first-principles recalculation of training data is requested, a corresponding ASE calculator must be assigned.'
            print(f'\tRecalculating ab initio energies and forces for new data points.')
            
            start_time_train = time.time()
            # print("\n\tAb initio calculations:")
            
            Nai = len(to_evaluate)
            for n,structure in enumerate(to_evaluate):
                print(f"\tAb initio calculations: {n+1:3}/{Nai:3}",end="\r")
                structure.calc = calculator_factory(n,None)
                structure.info = {}
                structure.arrays = {
                    "positions" : structure.get_positions(),
                    "numbers" : structure._get_atomic_numbers()
                }
                structure.get_potential_energy() # this will trigger the calculation
                results:dict = structure.calc.results
                for key,value in results.items():
                    if key in ['energy','free_energy','dipole','stress']:
                        structure.info[f"REF_{key}"] = value
                    elif key in ['forces']:
                        structure.arrays["REF_forces"] = value
                    else: 
                        structure.info[f"REF_{key}"] = value
                structure.calc = None
                # print(structure.info.keys())
                # structure.info["REF_energy"] = structure.info["energy"].pop()
                # structure.info["REF_forces"] = structure.info["forces"].pop()
                
            end_time_train = time.time()
        
            elapsed = end_time_train - start_time_train
            print(f'\n\tTime spent in ab initio calculations:   {elapsed:.2f} seconds')
            
        #-------------------------#
        # 3.b) Dataset updating
        #-------------------------#
        
        training_set.extend(to_evaluate)
        candidates = [item for i, item in enumerate(candidates) if i not in idcs_selected]
        
        # update the candidate file name
        new_candidates = f'{ofolder}/structures/candidates.n={iter}.extxyz'
        write(new_candidates, candidates, format='extxyz')
        fn_candidates = new_candidates
        
                
        #-------------------------#
        # 3.c) Dataset updating
        #-------------------------#
        
        # update the training set file name
        new_training_set = f'{ofolder}/structures/train-iter.n={iter}.extxyz'
        write(new_training_set, training_set, format='extxyz')
        shutil.copy(new_training_set, f'{ofolder}/train-iter.extxyz') # MACE will use this file to train
        
        #-------------------------#
        # 4) Models training
        #-------------------------#
        # retrain the committee with the enriched training set
        start_time_train = time.time()
        print(f'\tRetraining committee.')
        # TODO: add model refinement: check that it is actually done
        global GLOBAL_CONFIG_PATH
        GLOBAL_CONFIG_PATH = config
        
        if parallel: # parallel version
            with multiprocessing.Pool(processes=multiprocessing.cpu_count()) as pool:
                pool.map(train_single_model, range(n_committee))
                
        else: # serial version
            for n in range(n_committee):
                train_single_model(n)
        end_time_train = time.time()
        
        elapsed = end_time_train - start_time_train
        print(f'\ttraining duration:   {elapsed:.2f} seconds')
                
        # clean_output(ofolder,n_committee)
        
        #-------------------------#
        # Final messages
        #-------------------------#
        print(f'\n\tResults:')
        print(f'\t               Disagreement (pool): {avg_disagreement_pool:06f} eV')
        print(f'\t           Disagreement (selected): {avg_disagreement_selected:06f} eV')
        print(f'\t                New training set size: {len(training_set):d}')
        print(f'\t               New candidate set size: {len(candidates):d}')
        
        end_time = time.time()
        end_datetime = datetime.now()
        elapsed = end_time - start_time

        # print(f'\n\tEnd of QbC iteration {iter+1:d}/{n_iter:d}')
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
    
    os.remove(f'{ofolder}/train-iter.extxyz')
    
    return
  
#-------------------------#        
def copy_files_in_folder(src,dst):
    [shutil.copy(f"{src}/{f}", dst) for f in os.listdir(src) if os.path.isfile(f"{src}/{f}")]

#-------------------------#
@contextmanager
def timing(title="Duration"):
    start = time.time()
    yield
    end = time.time()
    print(f"\t{title}: {end - start:.2f}s")

#-------------------------#  
# def clean_output(folder,n_committee):
#     # remove useless files
#     for filename in os.listdir(f'{folder}/log'):
#         if filename.endswith('_debug.log'):
#             file_path = os.path.join(f'{folder}/log', filename)
#             os.remove(file_path)
            
#     for n in range(n_committee):
        
#         # models
#         filenames = [f"{folder}/models/mace.com={n}.model",
#                     f"{folder}/models/mace.com={n}_compiled.model",
#                     f"{folder}/models/mace.com={n}_stagetwo.model"]
#         for filename in filenames:
#             if os.path.exists(filename):
#                 os.remove(filename)
        
#         if os.path.exists(f"{folder}/models/mace.com={n}_stagetwo_compiled.model"):
#             os.rename(f"{folder}/models/mace.com={n}_stagetwo_compiled.model",f"{folder}/models/mace.n={n}.model")
        
#     for filename in os.listdir(f'{folder}/results'):
#         if filename.endswith('.txt') or filename.endswith('stage_one.png'):
#             file_path = os.path.join(f'{folder}/results', filename)
#             os.remove(file_path)

def prepare_train_file(template, output_path:str, replacements: dict):
    with open(template, 'r') as f:
        content = f.read()

    # Replace each key with its corresponding value
    for key, value in replacements.items():
        content = content.replace(key, str(value))

    with open(output_path, 'w') as f:
        f.write(content)