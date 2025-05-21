#!/bin/bash
set -e
source "${IPIPATH}/env.sh"

SOCKET=true
model_folder="../notebooks/init-train/models/"

if ${SOCKET}; then 
    echo "Starting i-PI socket server..."
    i-pi committee4nvt.ffsocket.xml &

    sleep 5  # or better: wait until socket files exist

    for n in {0..3}; do
        echo "Starting driver $n..."
        i-pi-py_driver -u -a address-${n} -m mace -o template=start.extxyz,model=${model_folder}/mace.n=${n}.model &
    done

    wait
else
    echo "Starting i-PI in direct mode..."
    i-pi committee4nvt.ffdirect.xml
fi
