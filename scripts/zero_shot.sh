#!/bin/bash

export PYTHONPATH=../
export CUDA_VISIBLE_DEVICES=1
id=59

repr_dims=256
seq_len=12
date=1201
sample_num=100
epochs=20

python ../exps/generate_task_feature.py --dataset PEMS-BAY/pems-bay $date --loader h5 --repr_dims $repr_dims --seq_len $seq_len --sample_num $sample_num --epochs $epochs --ratio 0.7 0.1 0.2
python ../exps/generate_task_feature.py --dataset traffic/traffic $date --loader csv --repr_dims $repr_dims --seq_len $seq_len --sample_num $sample_num --epochs $epochs --ratio 0.7 0.1 0.2
python ../exps/generate_task_feature.py --dataset electricity/electricity $date --loader csv --repr_dims $repr_dims --seq_len $seq_len --sample_num $sample_num --epochs $epochs --ratio 0.7 0.1 0.2
python ../exps/generate_task_feature.py --dataset radiation/solar-radiation $date --loader csv --repr_dims $repr_dims --seq_len $seq_len --sample_num $sample_num --epochs $epochs --ratio 0.34 0.33 0.33
python ../exps/generate_task_feature.py --dataset air_quality/air-pm2.5 $date --loader csv --repr_dims $repr_dims --seq_len $seq_len --sample_num $sample_num --epochs $epochs --ratio 0.34 0.33 0.33
python ../exps/generate_task_feature.py --dataset air_quality/air-pm10 $date --loader csv --repr_dims $repr_dims --seq_len $seq_len --sample_num $sample_num --epochs $epochs --ratio 0.34 0.33 0.33


repr_dims=256
seq_len=48
date=1201
sample_num=100
epochs=20

python ../exps/generate_task_feature.py --dataset PEMS-BAY/pems-bay $date --loader h5 --repr_dims $repr_dims --seq_len $seq_len --sample_num $sample_num --epochs $epochs --ratio 0.7 0.1 0.2
python ../exps/generate_task_feature.py --dataset traffic/traffic $date --loader csv --repr_dims $repr_dims --seq_len $seq_len --sample_num $sample_num --epochs $epochs --ratio 0.7 0.1 0.2
python ../exps/generate_task_feature.py --dataset electricity/electricity $date --loader csv --repr_dims $repr_dims --seq_len $seq_len --sample_num $sample_num --epochs $epochs --ratio 0.7 0.1 0.2
python ../exps/generate_task_feature.py --dataset radiation/solar-radiation $date --loader csv --repr_dims $repr_dims --seq_len $seq_len --sample_num $sample_num --epochs $epochs --ratio 0.34 0.33 0.33
python ../exps/generate_task_feature.py --dataset air_quality/air-pm2.5 $date --loader csv --repr_dims $repr_dims --seq_len $seq_len --sample_num $sample_num --epochs $epochs --ratio 0.34 0.33 0.33
python ../exps/generate_task_feature.py --dataset air_quality/air-pm10 $date --loader csv --repr_dims $repr_dims --seq_len $seq_len --sample_num $sample_num --epochs $epochs --ratio 0.34 0.33 0.33

sample_scale=200000
seq_len=12

python ../exps/random_search.py   --dataset PEMS-BAY/pems-bay --mode search --exp_id $id --sample_scale $sample_scale --seq_len $seq_len
python ../exps/random_search.py   --dataset traffic/traffic --mode search --exp_id $id --sample_scale $sample_scale --seq_len $seq_len
python ../exps/random_search.py   --dataset electricity/electricity --mode search --exp_id $id --sample_scale $sample_scale --seq_len $seq_len
python ../exps/random_search.py   --dataset radiation/solar-radiation --mode search --exp_id $id --sample_scale $sample_scale --seq_len $seq_len
python ../exps/random_search.py   --dataset air_quality/air-pm2.5 --mode search --exp_id $id --sample_scale $sample_scale --seq_len $seq_len
python ../exps/random_search.py   --dataset air_quality/air-pm10 --mode search --exp_id $id --sample_scale $sample_scale --seq_len $seq_len

seq_len=48

python ../exps/random_search.py   --dataset PEMS-BAY/pems-bay --mode search --exp_id $id --sample_scale $sample_scale --seq_len $seq_len
python ../exps/random_search.py   --dataset traffic/traffic --mode search --exp_id $id --sample_scale $sample_scale --seq_len $seq_len
python ../exps/random_search.py   --dataset electricity/electricity --mode search --exp_id $id --sample_scale $sample_scale --seq_len $seq_len
python ../exps/random_search.py   --dataset radiation/solar-radiation --mode search --exp_id $id --sample_scale $sample_scale --seq_len $seq_len
python ../exps/random_search.py   --dataset air_quality/air-pm2.5 --mode search --exp_id $id --sample_scale $sample_scale --seq_len $seq_len
python ../exps/random_search.py   --dataset air_quality/air-pm10 --mode search --exp_id $id --sample_scale $sample_scale --seq_len $seq_len

seq_len=12
CUDA_VISIBLE_DEVICES=1 python ../exps/generate_seeds.py  --dataset PEMS-BAY/pems-bay --datatype h5 --in_dim 1 --num_nodes 325 --seq_len $seq_len --mode train --epochs 100  --exp_id $id --ratio 0.7 0.1 0.2 &
CUDA_VISIBLE_DEVICES=4 python ../exps/generate_seeds.py  --dataset traffic/traffic --datatype csv --in_dim 1 --num_nodes 862 --seq_len $seq_len --mode train --epochs 100  --exp_id $id --ratio 0.7 0.1 0.2 &
CUDA_VISIBLE_DEVICES=5 python ../exps/generate_seeds.py  --dataset electricity/electricity --datatype csv --in_dim 1 --num_nodes 321 --seq_len $seq_len --mode train --epochs 100  --exp_id $id --ratio 0.7 0.1 0.2 &
CUDA_VISIBLE_DEVICES=6 python ../exps/generate_seeds.py  --dataset radiation/solar-radiation --datatype csv --in_dim 1 --num_nodes 50 --seq_len $seq_len --mode train --epochs 100  --exp_id $id --ratio 0.34 0.33 0.33 &
CUDA_VISIBLE_DEVICES=7 python ../exps/generate_seeds.py  --dataset air_quality/air-pm2.5 --datatype csv --in_dim 1 --num_nodes 12 --seq_len $seq_len --mode train --epochs 100  --exp_id $id --ratio 0.34 0.33 0.33 &
CUDA_VISIBLE_DEVICES=7 python ../exps/generate_seeds.py  --dataset air_quality/air-pm10 --datatype csv --in_dim 1 --num_nodes 12 --seq_len $seq_len --mode train --epochs 100  --exp_id $id --ratio 0.34 0.33 0.33 &

seq_len=48
CUDA_VISIBLE_DEVICES=1 python ../exps/generate_seeds.py  --dataset PEMS-BAY/pems-bay --datatype h5 --in_dim 1 --num_nodes 325 --seq_len $seq_len --mode train --epochs 100  --exp_id $id --ratio 0.7 0.1 0.2 &
CUDA_VISIBLE_DEVICES=4 python ../exps/generate_seeds.py  --dataset traffic/traffic --datatype csv --in_dim 1 --num_nodes 862 --seq_len $seq_len --mode train --epochs 100  --exp_id $id --ratio 0.7 0.1 0.2 &
CUDA_VISIBLE_DEVICES=5 python ../exps/generate_seeds.py  --dataset electricity/electricity --datatype csv --in_dim 1 --num_nodes 321 --seq_len $seq_len --mode train --epochs 100  --exp_id $id --ratio 0.7 0.1 0.2 &
CUDA_VISIBLE_DEVICES=6 python ../exps/generate_seeds.py  --dataset radiation/solar-radiation --datatype csv --in_dim 1 --num_nodes 50 --seq_len $seq_len --mode train --epochs 100  --exp_id $id --ratio 0.34 0.33 0.33 &
CUDA_VISIBLE_DEVICES=7 python ../exps/generate_seeds.py  --dataset air_quality/air-pm2.5 --datatype csv --in_dim 1 --num_nodes 12 --seq_len $seq_len --mode train --epochs 100  --exp_id $id --ratio 0.34 0.33 0.33 &
CUDA_VISIBLE_DEVICES=7 python ../exps/generate_seeds.py  --dataset air_quality/air-pm10 --datatype csv --in_dim 1 --num_nodes 12 --seq_len $seq_len --mode train --epochs 100  --exp_id $id --ratio 0.34 0.33 0.33 &

