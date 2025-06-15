#!/bin/bash

# Run IsolationForest for seeds 1-5
echo "Running training phase ..."
python run_benchmark.py --train \
    --subsample 20 \
    --benchmark gnn --epochs 20 \
    --learning-rate 0.001 --gnn-epochs 100 --hidden-size 64 --num-layers 3 \
    --use-wandb --wandb-name "gnn_subsample_20_train" --wandb-tags "gnn, train"
echo "Completed training"
echo "----------------------------------------"

echo "All training runs completed!"

# Run testing only for all seeds
echo "Running testing phase ..."
python run_benchmark.py --test \
    --subsample 20 \
    --benchmark gnn --epochs 20 \
    --learning-rate 0.001 --gnn-epochs 100 --hidden-size 64 --num-layers 3 \
    --use-wandb --wandb-name "gnn_subsample_20_test" --wandb-tags "gnn, test"
echo "Completed testing"
echo "----------------------------------------"
done

echo "All runs completed!" 


python run_benchmark.py --train \
    --subsample 100 \
    --benchmark gnn --epochs 20 \
    --learning-rate 0.001 --gnn-epochs 100 --hidden-size 64 --num-layers 3 \
    --use-wandb --wandb-name "gnn_subsample_100_train" --wandb-tags "gnn, train"