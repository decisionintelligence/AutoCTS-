#!/bin/bash

export PYTHONPATH=../
export CUDA_VISIBLE_DEVICES=7


python ../exps/dataset_slice.py --dataset pems/PEMS03 --datatype npz --num_nodes 358 --sample_num 80
python ../exps/dataset_slice.py --dataset pems/PEMS04 --datatype npz --num_nodes 307 --sample_num 80
python ../exps/dataset_slice.py --dataset pems/PEMS07 --datatype npz --num_nodes 883 --sample_num 80
python ../exps/dataset_slice.py --dataset pems/PEMS08 --datatype npz --num_nodes 170 --sample_num 80

python ../exps/dataset_slice.py --dataset PEMS-BAY/pems-bay --datatype h5 --num_nodes 325 --sample_num 80
python ../exps/dataset_slice.py --dataset METR-LA/metr-la --datatype h5 --num_nodes 207 --sample_num 80

python ../exps/dataset_slice.py --dataset ETT-small/ETTh1 --datatype csv --num_nodes 7 --sample_num 16
python ../exps/dataset_slice.py --dataset ETT-small/ETTh2 --datatype csv --num_nodes 7 --sample_num 16
python ../exps/dataset_slice.py --dataset ETT-small/ETTm1 --datatype csv --num_nodes 7 --sample_num 16
python ../exps/dataset_slice.py --dataset ETT-small/ETTm2 --datatype csv --num_nodes 7 --sample_num 16

python ../exps/dataset_slice.py --dataset exchange_rate/exchange_rate --datatype txt --num_nodes 8 --sample_num 16
python ../exps/dataset_slice.py --dataset solar/solar_AL --datatype txt --num_nodes 137 --sample_num 80
