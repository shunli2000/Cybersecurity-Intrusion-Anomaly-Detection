#!/bin/bash

# Run IsolationForest for seeds 1-5
echo "Running GNN with seed $seed"
python run_benchmark.py --train --benchmark gnn --subsample 100 --use_wandb --wandb_name "gnn_subsample_100" --wandb_tags "gnn"
echo "Completed seed $seed"
echo "----------------------------------------"

echo "All training runs completed!"

# Run testing only for all seeds
echo "Running testing phase for all seeds..."
python run_benchmark.py --test --benchmark gnn --subsample 100
echo "Completed testing"
echo "----------------------------------------"
done

echo "All runs completed!" 