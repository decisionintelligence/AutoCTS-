#!/bin/bash

#!/bin/bash

export PYTHONPATH=../
export CUDA_VISIBLE_DEVICES=0
id=59

#repr_dims=256
#seq_len=12
#date=1201
#sample_num=100
#epochs=20
#python ../exps/generate_task_feature.py --dataset PEMSD7M/PEMSD7M  $date --loader csv --repr_dims $repr_dims --seq_len $seq_len --sample_num $sample_num --epochs $epochs --ratio 0.6 0.2 0.2
#python ../exps/generate_task_feature.py --dataset NYC_TAXI/NYC_TAXI $date --loader PEMS --repr_dims $repr_dims --seq_len $seq_len --sample_num $sample_num --epochs $epochs --ratio 0.6 0.2 0.2
#python ../exps/generate_task_feature.py --dataset NYC_BIKE/NYC_BIKE $date --loader PEMS --repr_dims $repr_dims --seq_len $seq_len --sample_num $sample_num --epochs $epochs --ratio 0.6 0.2 0.2
#
#repr_dims=256
#seq_len=48
#date=1201
#sample_num=100
#epochs=20
#python ../exps/generate_task_feature.py --dataset PEMSD7M/PEMSD7M $date --loader csv --repr_dims $repr_dims --seq_len $seq_len --sample_num $sample_num --epochs $epochs --ratio 0.6 0.2 0.2
#python ../exps/generate_task_feature.py --dataset NYC_TAXI/NYC_TAXI $date --loader PEMS --repr_dims $repr_dims --seq_len $seq_len --sample_num $sample_num --epochs $epochs --ratio 0.6 0.2 0.2
#python ../exps/generate_task_feature.py --dataset NYC_BIKE/NYC_BIKE $date --loader PEMS --repr_dims $repr_dims --seq_len $seq_len --sample_num $sample_num --epochs $epochs --ratio 0.6 0.2 0.2

sample_scale=300000
seq_len=12
#python ../exps/random_search.py   --dataset PEMSD7M/PEMSD7M --mode search --exp_id $id --sample_scale $sample_scale --seq_len $seq_len
python ../exps/random_search.py   --dataset NYC_TAXI/NYC_TAXI --mode search --exp_id $id --sample_scale $sample_scale --seq_len $seq_len
python ../exps/random_search.py   --dataset NYC_BIKE/NYC_BIKE --mode search --exp_id $id --sample_scale $sample_scale --seq_len $seq_len
python ../exps/random_search.py   --dataset traffic/traffic --mode search --exp_id $id --sample_scale $sample_scale --seq_len $seq_len

seq_len=48

#python ../exps/random_search.py   --dataset PEMSD7M/PEMSD7M --mode search --exp_id $id --sample_scale $sample_scale --seq_len $seq_len
python ../exps/random_search.py   --dataset NYC_TAXI/NYC_TAXI --mode search --exp_id $id --sample_scale $sample_scale --seq_len $seq_len
python ../exps/random_search.py   --dataset NYC_BIKE/NYC_BIKE --mode search --exp_id $id --sample_scale $sample_scale --seq_len $seq_len
python ../exps/random_search.py   --dataset traffic/traffic --mode search --exp_id $id --sample_scale $sample_scale --seq_len $seq_len

seq_len=12
#CUDA_VISIBLE_DEVICES=0 python ../exps/generate_seeds.py  --dataset PEMSD7M/PEMSD7M --datatype csv --in_dim 1 --num_nodes 228 --seq_len $seq_len --mode train --epochs 100  --exp_id $id --ratio 0.6 0.2 0.2 &
CUDA_VISIBLE_DEVICES=0 python ../exps/generate_seeds.py  --dataset NYC_TAXI/NYC_TAXI --datatype npz --in_dim 2 --num_nodes 266 --seq_len $seq_len --mode train --epochs 100  --exp_id $id --ratio 0.6 0.2 0.2 &
CUDA_VISIBLE_DEVICES=2 python ../exps/generate_seeds.py  --dataset NYC_BIKE/NYC_BIKE --datatype npz --in_dim 2 --num_nodes 250 --seq_len $seq_len --mode train --epochs 100  --exp_id $id --ratio 0.6 0.2 0.2 &
CUDA_VISIBLE_DEVICES=4 python ../exps/generate_seeds.py  --dataset traffic/traffic --datatype csv --in_dim 1 --num_nodes 862 --seq_len $seq_len --mode train --epochs 100  --exp_id $id --ratio 0.7 0.1 0.2 &

seq_len=48
#CUDA_VISIBLE_DEVICES=0 python ../exps/generate_seeds.py  --dataset PEMSD7M/PEMSD7M --datatype csv --in_dim 1 --num_nodes 228 --seq_len $seq_len --mode train --epochs 100  --exp_id $id --ratio 0.6 0.2 0.2 &
CUDA_VISIBLE_DEVICES=6 python ../exps/generate_seeds.py  --dataset traffic/traffic --datatype csv --in_dim 1 --num_nodes 862 --seq_len $seq_len --mode train --epochs 100 --batch_size 32 --exp_id $id --ratio 0.7 0.1 0.2 &
CUDA_VISIBLE_DEVICES=2 python ../exps/generate_seeds.py  --dataset NYC_TAXI/NYC_TAXI --datatype npz --in_dim 2 --num_nodes 266 --seq_len $seq_len --mode train --epochs 100  --exp_id $id --ratio 0.6 0.2 0.2 &
CUDA_VISIBLE_DEVICES=0 python ../exps/generate_seeds.py  --dataset NYC_BIKE/NYC_BIKE --datatype npz --in_dim 2 --num_nodes 250 --seq_len $seq_len --mode train --epochs 100  --exp_id $id --ratio 0.6 0.2 0.2 &
