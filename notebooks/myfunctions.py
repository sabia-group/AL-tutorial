import os, sys, glob, time, shutil, multiprocessing
from datetime import datetime
import numpy as np
from typing import List, Dict, Union
from ase import Atoms
from ase.io import read, write
from ase.calculators.calculator import Calculator
from contextlib import redirect_stdout, redirect_stderr
from mace.cli.run_train import main as mace_run_train_main          # train a MACE model
from mace.cli.eval_configs import main as mace_eval_configs_main    # evaluate a MACE model

#-------------------------#
def extxyz2energy(file:str,keyword:str="MACE_energy"):
    """Extracts the energy values from an extxyz file and returns a numpy array
    """
    atoms = read(file, index=':')
    atoms = [ _correct_read(a) for a in atoms ]
    data = np.zeros(len(atoms),dtype=float)
    for n,atom in enumerate(atoms):
        data[n] = atom.info[keyword]
    return data

#-------------------------#
def extxyz2array(file:str,keyword:str="MACE_forces"):
    """Extracts the energy values from an extxyz file and returns a numpy array
    """
    atoms = read(file, index=':')
    atoms = [ _correct_read(a) for a in atoms ]
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
def eval_mace(model:str,infile:str,outfile:str,dtype:str="float32",batch_size:int=4):
    """Evaluate a MACE model.
    """
    sys.argv = ["program", 
                "--config", infile,
                "--output",outfile,
                "--model",model,
                "--default_dtype",dtype,
                "--batch_size",str(batch_size)]
    with open(os.devnull, 'w') as fnull:
        with redirect_stdout(fnull), redirect_stderr(fnull):
            mace_eval_configs_main()

#-------------------------#
def eval_and_extract(args):
    model, fn_in, fn_out, xtr = args
    eval_mace(model, fn_in, fn_out)
    if xtr:
        return extxyz2array(fn_out)
    else:
        return

#-------------------------#
def forces2disagreement(forces:np.ndarray)->np.ndarray:
    """Compute the atomic-level disagreement from a committee of MACE models."""
    
    forces = np.array(forces)  # Ensure forces is a numpy array

    # compute deviations from mean force (shape: [n_committee, n_samples, n_atoms, 3])
    dforces = forces - forces.mean(axis=0)[None, ...]

    # compute atomic-level disagreement (standard deviation of force norm)
    # - square, sum over x/y/z components -> norm²
    # - average over committee members
    # - take sqrt to get standard deviation per atom
    disagreement_atomic = np.sqrt(((dforces**2).sum(axis=3)).mean(axis=0))

    # average over atoms to get a single disagreement value per structure
    return disagreement_atomic.mean(axis=1)

#-------------------------#
def forces2rmse(forces: np.ndarray, ref_forces: np.ndarray) -> np.ndarray:
    """Compute RMSE per structure between averaged predicted and reference forces."""
    assert forces.ndim == 4                     # (committee, structures, atoms, 3)
    assert ref_forces.ndim == 3                 # (structures, atoms, 3)
    assert forces.shape[1:] == ref_forces.shape # shape check
    avg = forces.mean(axis=0)                   # avg over committee
    err2 = (avg - ref_forces)**2                # squared error
    mse  = err2.sum(axis=2).mean(axis=1)        # sum over components, avg over atoms
    return np.sqrt(mse)                         # RMSE per structure

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
        os.system(f"ulimit -s unlimited && {command} ")
        # if using mac os uncomment the following line
        # os.system(f"ulimit -s hard && {command} ")
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
def copy_files_in_folder(src,dst):
    [shutil.copy(f"{src}/{f}", dst) for f in os.listdir(src) if os.path.isfile(f"{src}/{f}")]

#-------------------------#
def prepare_train_file(template, output_path:str, replacements: dict):
    with open(template, 'r') as f:
        content = f.read()

    # Replace each key with its corresponding value
    for key, value in replacements.items():
        content = content.replace(key, str(value))

    with open(output_path, 'w') as f:
        f.write(content)

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
    Main Query-by-Committee (QbC) iterative training loop.

    Parameters
    ----------
    init_train_folder : str
        Folder containing initial training models and checkpoints.
    init_train_file : str
        File with initial training structures.
    fn_candidates : str
        File with candidate structures for evaluation.
    n_iter : int
        Total number of QbC iterations to perform.
    config : str
        Directory with MACE configuration files named config.0.yml, config.1.yml, etc.
    test_dataset : str, optional
        Dataset file for testing model performance (default is None).
    ofolder : str, optional
        Output folder to save QbC results (default "qbc-work").
    n_add_iter : int, optional
        Number of new candidates to add to training set per iteration (default 10).
    recalculate_selected : bool, optional
        If True, recompute energies and forces for selected structures using ASE calculator (default False).
    calculator_factory : callable, optional
        Factory function to create ASE calculators for recomputation (required if recalculate_selected=True).
    parallel : bool, optional
        Whether to train models in parallel (default True).
    """
    print(f'Starting QbC.')
    print(f"Please be sure that the config files use '{ofolder}/train-iter.extxyz' as training dataset.")
    
    if recalculate_selected:
        assert calculator_factory is not None, "Must provide ASE calculator factory to recalculate energies and forces."

    #-------------------------#
    # Prepare output directories
    #-------------------------#
    folders = [ofolder,
               f"{ofolder}/eval",
               f"{ofolder}/structures",
               f"{ofolder}/models",
               f"{ofolder}/checkpoints"]
    for f in folders:
        os.makedirs(f, exist_ok=True)
        
    #-------------------------#
    # Copy initial models and checkpoints to output folder
    #-------------------------#
    print(f"Copying checkpoints from '{init_train_folder}/checkpoints/' to '{ofolder}/checkpoints/'.")
    copy_files_in_folder(f"{init_train_folder}/checkpoints/", f"{ofolder}/checkpoints/")
    
    print(f"Copying models from '{init_train_folder}/models/' to '{ofolder}/models/'.")
    copy_files_in_folder(f"{init_train_folder}/models/", f"{ofolder}/models/")
    
    #-------------------------#
    # Display run summary banner
    #-------------------------#
    model_dir = os.path.join(ofolder, "models")
    n_committee = len(glob.glob(os.path.join(model_dir, "mace.com=*_compiled.model")))
    assert n_committee > 1, "Committee must contain more than one model."

    
    print(f"Number of models in committee: {n_committee:d}")
    print(f"Number of iterations: {n_iter:d}")
    print(f"Number of new candidates added per iteration: {n_add_iter:d}")
    
    print("\nPreparing files ...")
    print(f"Candidates pool: '{fn_candidates}'")
    print(f"Copying candidates pool from '{fn_candidates}' to '{ofolder}/candidates.start.extxyz'.")
    shutil.copy(fn_candidates, f'{ofolder}/candidates.start.extxyz')
    fn_candidates = f'{ofolder}/candidates.start.extxyz'
    
    if test_dataset is not None:
        print(f"Test dataset: '{test_dataset}'")
        print(f"Copying test dataset from '{test_dataset}' to '{ofolder}/structures/test.extxyz'.")
        shutil.copy(test_dataset, f'{ofolder}/structures/test.extxyz')
        test_dataset = f'{ofolder}/structures/test.extxyz'
    else:
        print("No test dataset provided")
    
    #-------------------------#
    # Load initial datasets
    #-------------------------#
    candidates: List[Atoms] = read(fn_candidates, index=':')
    training_set: List[Atoms] = read(init_train_file, index=':')
    progress_disagreement = [None]*n_iter
    
    #-------------------------#
    # Initialize model filenames in the committee
    #-------------------------#
    fns_committee = [f'{ofolder}/models/mace.com={n}_compiled.model' for n in range(n_committee)]
    
    #-------------------------#
    # Main QbC iterative loop
    #-------------------------#    
    for iter in range(n_iter):
        start_time = time.time()
        start_datetime = datetime.now()
        print(f'\n\t--------------------------------------------------------------------')
        print(f'\tStarting QbC iteration {iter+1:d}/{n_iter:d}')
        print(f'\n\t    Started at: {start_datetime.strftime("%Y-%m-%d %H:%M:%S")}')
        
        #-------------------------#
        # 1) Evaluate all candidates with each committee model
        #-------------------------# 
        print(f'\t    Evaluating committee disagreement across candidate pool.')
        start_time_train = time.time()
        forces = [None]* n_committee  # Store forces from each model
        
        if parallel:
            
            # train datasets
            args_list = [
                (model, fn_candidates, f"{ofolder}/eval/train.model={n}.iter={iter}.extxyz", True)
                for n, model in enumerate(fns_committee)
            ]
            with multiprocessing.Pool(processes=multiprocessing.cpu_count()) as pool:
                forces = pool.map(eval_and_extract, args_list)
                
            if test_dataset is not None:
                # test datasets
                args_list = [
                    (model, test_dataset, f"{ofolder}/eval/test.model={n}.iter={iter}.extxyz", False)
                    for n, model in enumerate(fns_committee)
                ]
                with multiprocessing.Pool(processes=multiprocessing.cpu_count()) as pool:
                    pool.map(eval_and_extract, args_list)
                    
        else:
            for n, model in enumerate(fns_committee):
                fn_dump = f"{ofolder}/eval/train.model={n}.iter={iter}.extxyz"
                eval_mace(model, fn_candidates, fn_dump)  # Evaluate model on candidates
                forces[n] = extxyz2array(fn_dump)
                
                if test_dataset is not None:
                    eval_mace(model, test_dataset, f"{ofolder}/eval/test.model={n}.iter={iter}.extxyz")
                
        end_time_train = time.time()
        elapsed = end_time_train - start_time_train
        print(f'\t    Evaluation duration: {elapsed:.2f} seconds')
                
        #-------------------------#
        # 2) Calculate disagreement among committee predictions
        #-------------------------# 
        disagreement = 1000*forces2disagreement(forces)  # Calculate atomic-level disagreement
        avg_disagreement_pool = disagreement.mean()
        print(f'\n\t    Disagreement (pool average): {avg_disagreement_pool:06f} meV/ang')

        # Select top candidates with highest disagreement
        print(f'\t    Selecting {n_add_iter:d} candidates with highest disagreement.')
        idcs_selected = np.argsort(disagreement)[-n_add_iter:]
        disagreement_selected = disagreement[idcs_selected]
        avg_disagreement_selected = disagreement_selected.mean()
        print(f'\n\t    Disagreement (selected candidates): {avg_disagreement_selected:06f} meV/ang')
        
        progress_disagreement[iter] = np.array([avg_disagreement_selected,
                                               avg_disagreement_pool,
                                               np.std(disagreement_selected),
                                               np.std(disagreement),
                                               len(training_set),
                                               len(candidates)])
        
        to_evaluate: List[Atoms] = [candidates[i] for i in idcs_selected]
        
        #-------------------------#
        # 3.a) Optional: Recalculate energies and forces using ASE calculator
        #-------------------------#
        if recalculate_selected:
            assert calculator_factory is not None, \
                'ASE calculator factory required for ab initio recalculation of selected data.'
            print(f'\t    Recalculating ab initio energies and forces for selected candidates.')
            
            start_time_train = time.time()
            Nai = len(to_evaluate)
            for n, structure in enumerate(to_evaluate):
                print(f"\t    Ab initio calculation {n+1:3}/{Nai:3}", end="\r")
                structure.calc = calculator_factory(n, None)
                structure.info = {}
                structure.arrays = {
                    "positions": structure.get_positions(),
                    "numbers": structure._get_atomic_numbers()
                }
                structure.get_potential_energy()  # Triggers calculation
                
                results: dict = structure.calc.results
                for key, value in results.items():
                    if key in ['energy', 'free_energy', 'dipole', 'stress']:
                        structure.info[f"REF_{key}"] = value
                    elif key == 'forces':
                        structure.arrays["REF_forces"] = value
                    else:
                        structure.info[f"REF_{key}"] = value
                structure.calc = None
                
            end_time_train = time.time()
            elapsed = end_time_train - start_time_train
            print(f'\n\t    Time spent on ab initio calculations: {elapsed:.2f} seconds')
            
        #-------------------------#
        # 3.b) Update training and candidate datasets
        #-------------------------#
        training_set.extend(to_evaluate)
        candidates = [item for i, item in enumerate(candidates) if i not in idcs_selected]
        
        new_candidates = f'{ofolder}/structures/candidates.n={iter}.extxyz'
        write(new_candidates, candidates, format='extxyz')
        fn_candidates = new_candidates
                
        #-------------------------#
        # 3.c) Save updated training set
        #-------------------------#
        new_training_set = f'{ofolder}/structures/train-iter.n={iter}.extxyz'
        write(new_training_set, training_set, format='extxyz')
        shutil.copy(new_training_set, f'{ofolder}/train-iter.extxyz')  # File used by MACE training
        
        #-------------------------#
        # 4) Retrain committee models with updated training set
        #-------------------------#
        start_time_train = time.time()
        print(f'\n\t    Retraining committee models.')
        
        global GLOBAL_CONFIG_PATH
        GLOBAL_CONFIG_PATH = config
        
        if parallel:
            with multiprocessing.Pool(processes=multiprocessing.cpu_count()) as pool:
                pool.map(train_single_model, range(n_committee))
        else:
            for n in range(n_committee):
                train_single_model(n)
                
        for n in range(n_committee):
            os.remove(f"{ofolder}/models/mace.com={n}.model")
        
        end_time_train = time.time()
        elapsed = end_time_train - start_time_train
        print(f'\t    Training duration: {elapsed:.2f} seconds')
        
        #-------------------------#
        # Summary of iteration results
        #-------------------------#
        end_time = time.time()
        end_datetime = datetime.now()
        elapsed = end_time - start_time

        print(f'\n\t    Ended iteration at:   {end_datetime.strftime("%Y-%m-%d %H:%M:%S")}')
        print(f'\t    Iteration duration: {elapsed:.2f} seconds')
        print(f'\n\t    Training set size now: {len(training_set):d}')
        print(f'\t    Candidate set size now: {len(candidates):d}')
        
        header = "\
selected-mean\n\
pool-mean\n\
selected-std\n\
pool-std\n\
training-set-size\n\
candidate-set-size\
"
        np.savetxt(f'{ofolder}/disagreement.txt', np.array(progress_disagreement[:iter+1]), header=header, fmt='%12.8f')
        print(f"\t    Updated '{ofolder}/disagreement.txt'")
        
        
    #-------------------------#
    # Finalize QbC process
    #-------------------------#
    print(f'\n\t--------------------------------------------------------------------')
    print(f'\tQbC loop finished.\n')
    print(f'\tFinal training set size: {len(training_set):d}')
    print(f'\tFinal candidate set size: {len(candidates):d}')
            
    # Save final training set
    write(f'{ofolder}/train-final.extxyz', training_set, format='extxyz')
    
    os.remove(f'{ofolder}/train-iter.extxyz')
    
    return

from ase.calculators.calculator import Calculator, all_changes, all_properties
import logging

_shared_logger = None  # Global singleton

def get_shared_logger(log_path='fhi_aims_calculator.log'):
    global _shared_logger
    if _shared_logger is None:
        logger = logging.getLogger("FHIaimsLogger")
        logger.setLevel(logging.INFO)
        logger.propagate = False  # Avoid duplicate output if root logger is also configured

        # Create handler only once
        handler = logging.FileHandler(log_path, mode='a')
        formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
        handler.setFormatter(formatter)
        logger.addHandler(handler)

        _shared_logger = logger
    return _shared_logger
    
class FHIaimsCalculator(Calculator):
    implemented_properties = ['energy', 'free_energy', 'forces', 'stress']

    # Shared logger
    logger = get_shared_logger()

    def __init__(self, aims_command, control_file, directory='.', output_path="aims.out", **kwargs):
        super().__init__(**kwargs)
        self.aims_command = aims_command
        self.control_file = control_file
        self.directory = directory
        self.output_path = output_path

    def calculate(self, atoms: Atoms = None, properties=all_properties, system_changes=all_changes):
        super().calculate(atoms, properties, system_changes)
        os.makedirs(self.directory, exist_ok=True)

        # Paths
        geom_path = os.path.join(self.directory, 'geometry.in')
        control_path = os.path.join(self.directory, 'control.in')
        output_path = os.path.join(self.directory, self.output_path)

        cmd = f"{self.aims_command} > {self.output_path} 2>/dev/null"
        self.logger.info(f"Running AIMS command: {cmd}")
        run_single_aims_structure(atoms, self.directory, cmd, self.control_file)

        # After run: check output
        if not os.path.exists(output_path) or os.path.getsize(output_path) == 0:
            self.logger.error(f"AIMS output not found or empty in {output_path}")
            raise RuntimeError(f"AIMS calculation failed in '{self.directory}'")

        try:
            output_atoms = read(output_path, format="aims-output")
        except Exception as e:
            self.logger.error(f"Failed to read AIMS output in {output_path}: {e}")
            raise RuntimeError(f"Failed to parse AIMS output: {e}")


        try:
            output_atoms = read(output_path, format="aims-output")
        except Exception as e:
            self.logger.error(f"Failed to parse output at {output_path}: {e}")
            raise e

        self.results = {
            "energy": output_atoms.get_potential_energy(),
            "free_energy": output_atoms.get_potential_energy(),
            "forces": output_atoms.get_forces(),
            "stress": np.zeros(6)
        }
        self.logger.info(f"Calculation completed in '{self.directory}'")