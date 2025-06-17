import argparse
from ase.io import read, write

def main():
    parser = argparse.ArgumentParser(description="Remove 'cell' or 'Lattice' info from extxyz file.")
    argv = {"metavar" : "\b",}
    parser.add_argument("-i","--input", **argv, required=True, help="Path to input .extxyz file")
    parser.add_argument("-o","--output", **argv, required=True,  help="Path to output file without cell info")
    
    args = parser.parse_args()
    
    structures = read(args.input, index=":")
    for atoms in structures:
        atoms.calc = None
        atoms.set_pbc(False)
        atoms.set_cell(None)
        
    write(args.output,structures)

if __name__ == "__main__":
    main()

