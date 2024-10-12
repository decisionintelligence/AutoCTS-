#!/bin/bash

export PYTHONPATH=../
export CUDA_VISIBLE_DEVICES=4

date=1123
repr_dims=256
sample_num=100
epochs=20
loader='subset'
train_script="../exps/generate_task_feature.py"



seq_len=12
base="pems/PEMS03"
dataset=("7" "19" "24" "28" "37" "64")

for set in "${dataset[@]}"; do
    python  $train_script --dataset "${base}_${set}" \
                          $date\
                          --loader $loader \
                          --repr_dims $repr_dims\
                          --seq_len $seq_len \
                          --epochs $epochs \
                          --sample_num $sample_num
done

seq_len=48
base="pems/PEMS03"
dataset=("1" "3" "17" "21" "23" "62")

for set in "${dataset[@]}"; do
    python  $train_script --dataset "${base}_${set}" \
                          $date\
                          --loader $loader \
                          --repr_dims $repr_dims\
                          --seq_len $seq_len \
                          --epochs $epochs \
                          --sample_num $sample_num
done


seq_len=12
base="pems/PEMS04"
dataset=("11" "14" "25" "32" "44" "53")

for set in "${dataset[@]}"; do
    python  $train_script --dataset "${base}_${set}" \
                          $date\
                          --loader $loader \
                          --repr_dims $repr_dims\
                          --seq_len $seq_len \
                          --epochs $epochs \
                          --sample_num $sample_num
done

seq_len=48
base="pems/PEMS04"
dataset=("10" "16" "20" "25" "33" "42")

for set in "${dataset[@]}"; do
    python  $train_script --dataset "${base}_${set}" \
                          $date\
                          --loader $loader \
                          --repr_dims $repr_dims\
                          --seq_len $seq_len \
                          --epochs $epochs \
                          --sample_num $sample_num
done


seq_len=12
base="pems/PEMS07"
dataset=("0" "8" "15" "25" "44" "66")

for set in "${dataset[@]}"; do
    python  $train_script --dataset "${base}_${set}" \
                          $date\
                          --loader $loader \
                          --repr_dims $repr_dims\
                          --seq_len $seq_len \
                          --epochs $epochs \
                          --sample_num $sample_num
done

seq_len=48
base="pems/PEMS07"
dataset=("5" "13" "35" "21" "28" "43")

for set in "${dataset[@]}"; do
    python  $train_script --dataset "${base}_${set}" \
                          $date\
                          --loader $loader \
                          --repr_dims $repr_dims\
                          --seq_len $seq_len \
                          --epochs $epochs \
                          --sample_num $sample_num
done


seq_len=12
base="pems/PEMS08"
dataset=("5" "19" "25" "46" "61" "70")

for set in "${dataset[@]}"; do
    python  $train_script --dataset "${base}_${set}" \
                          $date\
                          --loader $loader \
                          --repr_dims $repr_dims\
                          --seq_len $seq_len \
                          --epochs $epochs \
                          --sample_num $sample_num
done

seq_len=48
base="pems/PEMS08"
dataset=("4" "10" "25" "30" "41" "64")

for set in "${dataset[@]}"; do
    python  $train_script --dataset "${base}_${set}" \
                          $date\
                          --loader $loader \
                          --repr_dims $repr_dims\
                          --seq_len $seq_len \
                          --epochs $epochs \
                          --sample_num $sample_num
done

seq_len=12
base="METR-LA/metr-la"
dataset=("1" "8" "26" "30" "45" "62")

for set in "${dataset[@]}"; do
    python  $train_script --dataset "${base}_${set}" \
                          $date\
                          --loader $loader \
                          --repr_dims $repr_dims\
                          --seq_len $seq_len \
                          --epochs $epochs \
                          --sample_num $sample_num
done

seq_len=48
base="METR-LA/metr-la"
dataset=("5" "9" "2" "34" "41" "63")

for set in "${dataset[@]}"; do
    python  $train_script --dataset "${base}_${set}" \
                          $date\
                          --loader $loader \
                          --repr_dims $repr_dims\
                          --seq_len $seq_len \
                          --epochs $epochs \
                          --sample_num $sample_num
done


seq_len=12
base="solar/solar_AL"
dataset=("1" "8" "25" "47" "62" "73")

for set in "${dataset[@]}"; do
    python  $train_script --dataset "${base}_${set}" \
                          $date\
                          --loader $loader \
                          --repr_dims $repr_dims\
                          --seq_len $seq_len \
                          --epochs $epochs \
                          --sample_num $sample_num
done

seq_len=48
base="solar/solar_AL"
dataset=("4" "13" "21" "46" "47" "61")

for set in "${dataset[@]}"; do
    python  $train_script --dataset "${base}_${set}" \
                          $date\
                          --loader $loader \
                          --repr_dims $repr_dims\
                          --seq_len $seq_len \
                          --epochs $epochs \
                          --sample_num $sample_num
done

seq_len=12
base="ETT-small/ETTh1"
dataset=("3" "6" "9")

for set in "${dataset[@]}"; do
    python  $train_script --dataset "${base}_${set}" \
                          $date\
                          --loader $loader \
                          --repr_dims $repr_dims\
                          --seq_len $seq_len \
                          --epochs $epochs \
                          --sample_num $sample_num
done

seq_len=48
base="ETT-small/ETTh1"
dataset=("2" "5" "13")

for set in "${dataset[@]}"; do
    python  $train_script --dataset "${base}_${set}" \
                          $date\
                          --loader $loader \
                          --repr_dims $repr_dims\
                          --seq_len $seq_len \
                          --epochs $epochs \
                          --sample_num $sample_num
done


seq_len=12
base="ETT-small/ETTh2"
dataset=("2" "7" "15")

for set in "${dataset[@]}"; do
    python  $train_script --dataset "${base}_${set}" \
                          $date\
                          --loader $loader \
                          --repr_dims $repr_dims\
                          --seq_len $seq_len \
                          --epochs $epochs \
                          --sample_num $sample_num
done

seq_len=48
base="ETT-small/ETTh2"
dataset=("3" "8" "13")

for set in "${dataset[@]}"; do
    python  $train_script --dataset "${base}_${set}" \
                          $date\
                          --loader $loader \
                          --repr_dims $repr_dims\
                          --seq_len $seq_len \
                          --epochs $epochs \
                          --sample_num $sample_num
done


seq_len=12
base="ETT-small/ETTm1"
dataset=("2" "5" "14")

for set in "${dataset[@]}"; do
    python  $train_script --dataset "${base}_${set}" \
                          $date\
                          --loader $loader \
                          --repr_dims $repr_dims\
                          --seq_len $seq_len \
                          --epochs $epochs \
                          --sample_num $sample_num
done

seq_len=48
base="ETT-small/ETTm1"
dataset=("3" "7" "15")

for set in "${dataset[@]}"; do
    python  $train_script --dataset "${base}_${set}" \
                          $date\
                          --loader $loader \
                          --repr_dims $repr_dims\
                          --seq_len $seq_len \
                          --epochs $epochs \
                          --sample_num $sample_num
done


seq_len=12
base="ETT-small/ETTm2"
dataset=("1" "11" "13")

for set in "${dataset[@]}"; do
    python  $train_script --dataset "${base}_${set}" \
                          $date\
                          --loader $loader \
                          --repr_dims $repr_dims\
                          --seq_len $seq_len \
                          --epochs $epochs \
                          --sample_num $sample_num
done

seq_len=48
base="ETT-small/ETTm2"
dataset=("1" "8" "15")

for set in "${dataset[@]}"; do
    python  $train_script --dataset "${base}_${set}" \
                          $date\
                          --loader $loader \
                          --repr_dims $repr_dims\
                          --seq_len $seq_len \
                          --epochs $epochs \
                          --sample_num $sample_num
done


seq_len=12
base="exchange_rate/exchange_rate"
dataset=("7" "10")

for set in "${dataset[@]}"; do
    python  $train_script --dataset "${base}_${set}" \
                          $date\
                          --loader $loader \
                          --repr_dims $repr_dims\
                          --seq_len $seq_len \
                          --epochs $epochs \
                          --sample_num $sample_num
done

seq_len=48
base="exchange_rate/exchange_rate"
dataset=("4" "11")

for set in "${dataset[@]}"; do
    python  $train_script --dataset "${base}_${set}" \
                          $date\
                          --loader $loader \
                          --repr_dims $repr_dims\
                          --seq_len $seq_len \
                          --epochs $epochs \
                          --sample_num $sample_num
done