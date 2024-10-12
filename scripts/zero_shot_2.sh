#!/bin/bash

export PYTHONPATH=../
export CUDA_VISIBLE_DEVICES=0
sample_scale=200000
seq_len=12
id=3


CUDA_VISIBLE_DEVICES=4 python ../exps/random_search_1.py   --dataset los_loop/los_speed --mode search --exp_id $id --sample_scale $sample_scale --seq_len $seq_len
CUDA_VISIBLE_DEVICES=1 python ../exps/random_search_1.py   --dataset sz_taxi/sz_speed --mode search --exp_id $id --sample_scale $sample_scale --seq_len $seq_len
CUDA_VISIBLE_DEVICES=2 python ../exps/random_search_1.py   --dataset NYC_BIKE/NYC_BIKE --mode search --exp_id $id --sample_scale $sample_scale --seq_len $seq_len
CUDA_VISIBLE_DEVICES=3 python ../exps/random_search_1.py   --dataset NYC_TAXI/NYC_TAXI --mode search --exp_id $id --sample_scale $sample_scale --seq_len $seq_len
#CUDA_VISIBLE_DEVICES=4 python ../exps/random_search_1.py   --dataset PEMSD7M/PEMSD7M --mode search --exp_id $id --sample_scale $sample_scale --seq_len $seq_len


#CUDA_VISIBLE_DEVICES=0 python ../exps/generate_seeds.py  --dataset PEMSD7M/PEMSD7M --datatype csv --in_dim 1 --num_nodes 228 --seq_len $seq_len --mode train --epochs 100  --exp_id $id --ratio 0.6 0.2 0.2 &
CUDA_VISIBLE_DEVICES=1 python ../exps/generate_seeds.py  --dataset NYC_TAXI/NYC_TAXI --datatype npz --in_dim 2 --num_nodes 266 --seq_len $seq_len --mode train --epochs 100  --exp_id $id --ratio 0.6 0.2 0.2 &
CUDA_VISIBLE_DEVICES=2 python ../exps/generate_seeds.py  --dataset NYC_BIKE/NYC_BIKE --datatype npz --in_dim 2 --num_nodes 250 --seq_len $seq_len --mode train --epochs 100  --exp_id $id --ratio 0.6 0.2 0.2 &
CUDA_VISIBLE_DEVICES=3 python ../exps/generate_seeds.py  --dataset los_loop/los_speed --datatype csv --in_dim 1 --num_nodes 207 --seq_len $seq_len --mode train --epochs 100  --exp_id $id --ratio 0.7 0.1 0.2 &
CUDA_VISIBLE_DEVICES=4 python ../exps/generate_seeds.py  --dataset sz_taxi/sz_speed --datatype csv --in_dim 1 --num_nodes 156 --seq_len $seq_len --mode train --epochs 100  --exp_id $id --ratio 0.7 0.1 0.2 &
