#!/bin/bash

model_dir=$1/serial_wf/model/
mkdir -p ${model_dir}

python ../rct-scripts/new_serial_multiphase_rct_v3.py \
	--num_phases 3 \
        --num_epochs 200 \
        --train_inner_iter 13 \
        --sim_inner_iter 8 \
        --data_root_dir $1/serial_wf/data/ \
        --model_dir ${model_dir} \
        --mat_size 3000 \
        --num_mult 384 \
        --num_allreduce 1 \
        --sim_rank 128 \
        --train_rank 16 \
        --train_preprocess_time 10 \
        --sim_read_size 42949672960 \
        --sim_write_size 10737418240 \
        --train_read_size 3221225472 \
        --project_id RECUP
