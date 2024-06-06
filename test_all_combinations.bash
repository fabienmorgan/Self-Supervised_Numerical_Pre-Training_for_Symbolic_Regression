#!/bin/bash

# Test path
test_path="data/benchmark/train_nc"

# Starting point
start=-10

# Loop over the model types
#for model_type in "mmsr_se1_oldloss" "mmsr_se5_oldloss" "mmsr_se5_newloss"; do


for model_type in "self_trained_nsr"; do
    # Loop over the range of values
    for ((i=0; i<=40; i++)); do
        # Calculate min_support based on i
        if ((i % 2 == 0)); then
            min_support=$((start - i / 2))
        else
            min_support=$((start + (i + 1) / 2))
        fi

        # Calculate max_support as min_support + 20
        max_support=$((min_support + 20))

        # Loop over the seed values
        for ((seed=1; seed<=10; seed++)); do
            # Run the Python script with the current values of min_support, max_support and seed
            python scripts/test.py --model_type $model_type --min_support $min_support --max_support $max_support --test_path $test_path --seed $seed --number_of_samples 500
        done
    done
done


for model_type in "self_trained_nsr"; do
    # Loop over the number of samples
    for number_of_samples in 100 250 500 1000 2000; do
        # Loop over the seed values
        for ((seed=1; seed<=10; seed++)); do
            # Run the Python script with the current values of model_type, number_of_samples, and seed
            python scripts/test.py --model_type $model_type --test_path $test_path --seed $seed --number_of_samples $number_of_samples
        done
    done
done


for model_type in "self_trained_nsr"; do
    # Loop over the range of values
    for min_support in -10 -5 -2.5 -1; do
        max_support=$(echo "$min_support * -1" | bc)

        # Loop over the seed values
        for ((seed=1; seed<=10; seed++)); do
            # Run the Python script with the current values of min_support, max_support and seed
            python scripts/test.py --model_type $model_type --min_support $min_support --max_support $max_support --test_path $test_path --seed $seed --number_of_samples 500
        done
    done
done
