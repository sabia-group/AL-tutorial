from myfunctions import run_qbc        # run Query by Committee
import numpy as np

# MACE committee
n_init_train = 20   # number of initial training structures
n_test       = 50   # number of test structures
n_committee  = 4    # number of MACE committee members
parallel     = True # whether to parallelize the training (this should work for Linux)

# QbC
n_iter_qbc = 10     # number of QbC iterations
n_add_iter = 10     # number of structures to add in each QbC iteration

# seeds to generate different MACE models
seeds = np.random.randint(0, 2**32 - 1, size=n_committee, dtype=np.uint32)

# folders
init_train_folder = "init-train"        # working folder for initial training
qbc_folder        = "qbc-work"          # working folder for QbC
md_folder         = "md"                # working folder for MD simulations

# run Query by Committee (QbC) to iteratively select structures
run_qbc(
    init_train_folder=init_train_folder,
    init_train_file=f"{init_train_folder}/structures/init.train.extxyz", # initial training dataset
    fn_candidates=f'{init_train_folder}/structures/candidates.extxyz',   # candidate structures
    test_dataset=f'{init_train_folder}/structures/test.extxyz',          # test set
    n_iter=n_iter_qbc,                                                   # number of QbC iterations
    config=f'{qbc_folder}/config',                                       # folder with config files
    ofolder=qbc_folder,                                                  # folder to save the QBC results
    n_add_iter=n_add_iter,                                               # number of structures to add in each iteration
    recalculate_selected=False,                                          # whether to recalculate the selected structures with DFT (part 2)
    parallel=parallel
);
# it should take 9m