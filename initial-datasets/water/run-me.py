from ase.io import read
for n in range(8):
    print(f"\n\tReading 'train.{n}.extxyz':")
    atoms = read(f"train.{n}.extxyz",index=":")
    print("\tn. of structures: ", len(atoms))
    print("\t     n. of atoms: ", atoms[0].get_global_number_of_atoms())
    print("\t            info: ", list(atoms[0].info.keys()))
    print("\t          arrays: ", list(atoms[0].arrays.keys()))
    print()