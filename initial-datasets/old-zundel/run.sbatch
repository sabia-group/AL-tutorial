#!/bin/bash

#SBATCH --partition=normal
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=72
#SBATCH --export=NONE
#SBATCH --time=00:20:00


# prepare environment
module load cp2k
module list

ulimit -s unlimited
export OMP_NUM_THREADS=1

# prepare file names
log='cp2k.out'

# print some information
echo ' started at:' `date`
echo '   hostname:' `hostname`
echo

# run
srun cp2k.psmp zundel.inp >> $log

# report finish time
echo 'finished at:' `date`
