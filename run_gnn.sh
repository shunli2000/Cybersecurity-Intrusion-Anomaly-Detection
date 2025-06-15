
python run_benchmark.py --train --test \
    --subsample 20 \
    --benchmark gnn --epochs 20 \
    --learning-rate 0.001 --gnn-epochs 100 --hidden-size 64 --num-layers 3 --k 5 \
    --use-wandb --wandb-name "gnn_subsample_20" --wandb-tags "gnn, train"


python run_benchmark.py --train --test \
    --benchmark ifor --epochs 1 \
    --outliers-fraction 0.3 \
    --use-wandb --wandb-name "ifor" --wandb-tags "gnn, train"