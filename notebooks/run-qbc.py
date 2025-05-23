# %%
import os, glob, re
import multiprocessing
from tqdm.notebook import tqdm
from IPython.display import Image, display

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

from ase.io import read, write # read and write structures

# import functions to run this tutorial
from myfunctions import train_mace     # train MACE model
from myfunctions import eval_mace      # evaluate MACE model
from myfunctions import extxyz2energy  # extract energy from extxyz file
from myfunctions import extxyz2forces  # extract forces from extxyz file
from myfunctions import run_qbc        # run Query by Committee

# %%
n_init_train = 20
n_test = 50  
n_committee = 4
parallel = False
md_folder = "md"
init_train_folder = "init-train"
qbc_folder = "qbc-work" # if you modify this, add the new folder to .gitignore
n_iter_qbc = 10
n_add_iter = 20     
seeds = np.random.randint(0, 2**32 - 1, size=n_committee, dtype=np.uint32)

# %%
#np.random.seed(0)
os.makedirs(f'{init_train_folder}', exist_ok=True)
os.makedirs(f'{init_train_folder}/config', exist_ok=True)
os.makedirs(f'{init_train_folder}/models', exist_ok=True)
os.makedirs(f'{init_train_folder}/eval', exist_ok=True)
os.makedirs(f'{init_train_folder}/structures', exist_ok=True)
os.makedirs(f'{md_folder}', exist_ok=True)

# %%
# Define different values for each config
# TODO: make this simpler - the only thing we need to change is the name of the training extxyz file.
os.makedirs(qbc_folder, exist_ok=True)
os.makedirs(f'{qbc_folder}/config', exist_ok=True)
# seeds = np.random.randint(0, 2**32 - 1, size=n_committee, dtype=np.uint32)
for i in range(n_committee):
    filename = f"{qbc_folder}/config/config.{i}.yml"
    name = f"mace.com={i}"
    
    config_text = f"""
# You can modify the following parameters
num_channels: 16
max_L: 0            # take it larger but not smaller
max_ell: 1          # take it larger but not smaller
correlation: 1      # take it larger but not smaller
num_interactions: 2 # take it larger but not smaller

# ... but you can also modify these ones
r_max: 4.0
batch_size: 4
max_num_epochs: 10000 # this is basically early stopping
patience: 10       # we are a bit in a rush

# But please, do not modify these parameters!
model: "MACE"
name: "{name}"

model_dir      : "{qbc_folder}/models"
log_dir        : "{qbc_folder}/log"
checkpoints_dir: "{qbc_folder}/checkpoints"
results_dir    : "{qbc_folder}/results"

train_file: "{qbc_folder}/train-iter.extxyz"
energy_key: "REF_energy"
forces_key: "REF_forces"
energy_weight: 1
forces_weight: 10

E0s: 
  1: -13.7487804074635
  8: -2045.41865185226
device: cpu
swa: false
seed: {seeds[i]}
restart_latest: True

"""

    with open(filename, "w") as f:
        f.write(config_text)

    print(f"Wrote {filename}")

# %%
# Attention: this function will not
run_qbc(
    init_train_folder=init_train_folder,
    init_train_file=f"{init_train_folder}/structures/init.train.extxyz", # initial training dataset
    fn_candidates=f'{init_train_folder}/structures/remaining.extxyz',    # candidate structures
    test_dataset=f'{init_train_folder}/structures/test.extxyz',          # test set
    n_iter=n_iter_qbc,                                                   # number of QbC iterations
    config=f'{qbc_folder}/config',                                       # folder with config files
    ofolder=qbc_folder,                                                  # folder to save the QBC results
    n_add_iter=n_add_iter,                                               # number of structures to add in each iteration
    recalculate_selected=False,                                          # whether to recalculate the selected structures with DFT (part 2)
    parallel=False
);