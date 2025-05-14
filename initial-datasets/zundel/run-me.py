from ase.io import read
print("\n\tReading 'train.extxyz':\n")
atoms = read("train.extxyz",index=":")
print("\tn. of structures: ", len(atoms))
print("\t     n. of atoms: ", atoms[0].get_global_number_of_atoms())
print("\t            info: ", list(atoms[0].info.keys()))
print("\t          arrays: ", list(atoms[0].arrays.keys()))
print()