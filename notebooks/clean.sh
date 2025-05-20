#!/bin/bash
foldeers=("__pycache__" "checkpoints" "config" "log" "models" "qbc-work" "results" "structures")
for folder in "${foldeers[@]}"; do
    rm -rf "${folder}"
done
