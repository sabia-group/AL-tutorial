#!/bin/bash
set -e
export OMP_NUM_THREADS=4
source "${IPIPATH}/env.sh"

SOCKET=true
model_folder="../../checkpoints/models"

if ${SOCKET}; then # parallel: this should be faster because the MACE models are run in parallel
    echo "Starting i-PI socket server..."
    i-pi committee4nvt.ffsocket.xml &

    sleep 5  # or better: wait until socket files exist

    for n in {0..3}; do
        echo "Starting driver $n..."
        i-pi-py_driver -u -a address-${n} -m mace -o template=start.extxyz,model=${model_folder}/mace.com=${n}_compiled.model &
    done

    wait
else # serial
    echo "Starting i-PI in direct mode..."
    i-pi committee4nvt.ffdirect.xml
fi

python ../post-process.py -i ipi.pos_0.extxyz -o zundel-inference.extxyz
