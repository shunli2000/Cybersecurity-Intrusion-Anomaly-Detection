#!/bin/bash

# Run IsolationForest for seeds 1-5
echo "Running IsolationForest with seed $seed"
python run_benchmark.py --train --benchmark ifor --epochs 1
echo "Completed seed $seed"
echo "----------------------------------------"

echo "All training runs completed!"

# Run testing only for all seeds
echo "Running testing phase for all seeds..."
python run_benchmark.py --test --benchmark ifor
echo "Completed testing"
echo "----------------------------------------"
done

echo "All runs completed!" 