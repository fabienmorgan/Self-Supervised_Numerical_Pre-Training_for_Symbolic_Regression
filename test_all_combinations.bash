#!/bin/bash

# Test path
test_path="data/benchmark/train_nc"

# Loop over the model types
for model_type in "mmsr" "nsr"; do
    # Loop over the range of values
    for ((i=-30; i<=10; i++)); do
        # Calculate max_support as min_support + 20
        j=$((i + 20))

        # Run the Python script with the current values of min_support and max_support
        python scripts/test.py --model_type $model_type --min_support $i --max_support $j --test_path $test_path
    done
done