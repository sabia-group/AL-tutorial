#!/bin/bash
foldeers=("__pycache__" "checkpoints" "config" "log" "models" "qbc-work" "results" "structures" "eval")
for folder in "${foldeers[@]}"; do
    rm -rf "${folder}"
done
