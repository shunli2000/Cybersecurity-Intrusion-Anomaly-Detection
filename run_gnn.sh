#!/bin/bash

# Run IsolationForest for seeds 1-5
echo "Running training phase ..."
python run_benchmark.py --train \
    --benchmark gnn --batch-size 128 --num-workers 16 \
    --subsample 100 \
    --use-wandb --wandb-name "gnn_subsample_100_train" --wandb-tags "gnn, train"
echo "Completed training"
echo "----------------------------------------"

echo "All training runs completed!"

# Run testing only for all seeds
echo "Running testing phase ..."
python run_benchmark.py --test --benchmark gnn --subsample 100 --use_wandb --wandb_name "gnn_subsample_100_test" --wandb_tags "gnn" "test"
echo "Completed testing"
echo "----------------------------------------"
done

echo "All runs completed!" 