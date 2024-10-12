#!/bin/bash
export PYTHONPATH=../
export CUDA_VISIBLE_DEVICES=2
repr_dims=256
seq_len=12
date=1127
sample_num=100
epochs=20

python ../exps/generate_task_feature.py --dataset pems/PEMS03 $date --loader PEMS --repr_dims $repr_dims --seq_len $seq_len --sample_num $sample_num --epochs $epochs
python ../exps/generate_task_feature.py --dataset pems/PEMS04 $date --loader PEMS --repr-dims $repr_dims --seq_len $seq_len --sample_num $sample_num --epochs $epochs
python ../exps/generate_task_feature.py --dataset pems/PEMS07 $date --loader PEMS --repr-dims $repr_dims --seq_len $seq_len --sample_num $sample_num --epochs $epochs
python ../exps/generate_task_feature.py --dataset pems/PEMS08 $date --loader PEMS --repr-dims $repr_dims --seq_len $seq_len --sample_num $sample_num --epochs $epochs

python ../exps/generate_task_feature.py --dataset METR-LA/metr-la $date --loader h5 --repr-dims $repr_dims --seq_len $seq_len --sample_num $sample_num --epochs $epochs
python ../exps/generate_task_feature.py --dataset solar/solar_AL $date --loader txt --repr-dims $repr_dims --seq_len $seq_len --sample_num $sample_num --epochs $epochs

python ../exps/generate_task_feature.py --dataset exchange_rate/exchange_rate $date --loader txt --repr_dims $repr_dims --seq_len $seq_len --sample_num $sample_num

python ../exps/generate_task_feature.py --dataset ETT-small/ETTh1 $date --loader ETT --repr-dims $repr_dims --seq_len $seq_len --sample_num $sample_num
python ../exps/generate_task_feature.py --dataset ETT-small/ETTm1 $date --loader ETT --repr-dims $repr_dims --seq_len $seq_len --sample_num $sample_num
python ../exps/generate_task_feature.py --dataset ETT-small/ETTh2 $date --loader ETT --repr-dims $repr_dims --seq_len $seq_len --sample_num $sample_num
python ../exps/generate_task_feature.py --dataset ETT-small/ETTm2 $date --loader ETT --repr-dims $repr_dims --seq_len $seq_len --sample_num $sample_num