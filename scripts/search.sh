#!/bin/bash

export PYTHONPATH=../
export CUDA_VISIBLE_DEVICES=0
id=0
sample_scale=300000


seq_len=12
python ../exps/random_search.py   --dataset PEMS-BAY/pems-bay --mode search --exp_id $id --sample_scale $sample_scale --seq_len $seq_len
python ../exps/random_search.py   --dataset traffic/traffic --mode search --exp_id $id --sample_scale $sample_scale --seq_len $seq_len
python ../exps/random_search.py   --dataset electricity/electricity --mode search --exp_id $id --sample_scale $sample_scale --seq_len $seq_len
python ../exps/random_search.py   --dataset PEMSD7M/PEMSD7M --mode search --exp_id $id --sample_scale $sample_scale --seq_len $seq_len
python ../exps/random_search.py   --dataset NYC_TAXI/NYC_TAXI --mode search --exp_id $id --sample_scale $sample_scale --seq_len $seq_len
python ../exps/random_search.py   --dataset NYC_BIKE/NYC_BIKE --mode search --exp_id $id --sample_scale $sample_scale --seq_len $seq_len

seq_len=48
python ../exps/random_search.py   --dataset PEMS-BAY/pems-bay --mode search --exp_id $id --sample_scale $sample_scale --seq_len $seq_len
python ../exps/random_search.py   --dataset traffic/traffic --mode search --exp_id $id --sample_scale $sample_scale --seq_len $seq_len
python ../exps/random_search.py   --dataset electricity/electricity --mode search --exp_id $id --sample_scale $sample_scale --seq_len $seq_len
python ../exps/random_search.py   --dataset PEMSD7M/PEMSD7M --mode search --exp_id $id --sample_scale $sample_scale --seq_len $seq_len
python ../exps/random_search.py   --dataset NYC_TAXI/NYC_TAXI --mode search --exp_id $id --sample_scale $sample_scale --seq_len $seq_len
python ../exps/random_search.py   --dataset NYC_BIKE/NYC_BIKE --mode search --exp_id $id --sample_scale $sample_scale --seq_len $seq_len
