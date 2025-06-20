#!/bin/bash
set -euo pipefail
source "${IPIPATH}/env.sh"

echo "Starting i-PI socket server..."
stdbuf -oL -eL i-pi input.xml > ipi.log 2>&1 &

# Optional: wait until i-PI socket is ready instead of fixed sleep
# Use a more robust method, e.g., checking for socket existence
# This is a placeholder sleep; update with actual condition if needed
sleep 5

# Launch all drivers
for n in {0..3}; do
    echo "Starting driver $n..."
    i-pi-py_driver -u -a address-${n} -m mace -o template=start.extxyz,model=mace.com=${n}_compiled.model &
done

# Wait for all background processes (drivers) to finish
wait

# Post-processing
# echo "Running post-processing..."
# python ../post-process.py -i ipi.pos_0.extxyz -o eigen-inference.extxyz
