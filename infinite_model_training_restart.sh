#!/bin/bash

# Trap the keyboard interrupt signal (SIGINT), so the script can be terminated with Ctrl+C
trap "echo 'Script terminated'; exit" SIGINT

while true; do
    bsub -I -gpu "num=2:mode=exclusive_process:mps=yes" python3 scripts/train.py
    echo "Command crashed with exit code $?. Respawning.." >&2
    sleep 1
done