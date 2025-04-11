#!/bin/bash

model_dir=$1/serial_wf/model/
mkdir -p ${model_dir}

python ../rct-scripts/new_serial_multiphase_rct_v3.py \
        --num_phases 3 \
        --num_epochs 100 \
        --train_inner_iter 3 \
        --sim_inner_iter 2 \
        --data_root_dir $1/serial_wf/data/ \
        --model_dir ${model_dir} \
        --mat_size 3000 \
        --num_mult 3072 \
        --num_allreduce 1 \
        --sim_rank 1024 \
        --train_rank 128 \
        --train_preprocess_time 10 \
        --sim_read_size 85899345920 \
        --sim_write_size 21474836480 \
        --train_read_size 206158430208 \
        --project_id RECUP
