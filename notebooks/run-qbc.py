from myfunctions import run_qbc        # run Query by Committee

# %%
n_init_train = 20
n_test = 50  
n_committee = 4
parallel = True
init_train_folder = "init-train"
qbc_folder = "qbc-work" 
n_iter_qbc = 10
n_add_iter = 20     

# %%
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
    parallel=parallel
)