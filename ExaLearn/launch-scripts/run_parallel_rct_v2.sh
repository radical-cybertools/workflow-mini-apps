#!/bin/bash

model_dir=$1/parallel_wf/model/
mkdir -p ${model_dir}

python ../rct-scripts/new_parallel_multiphase_rct_v2.py \
	--num_phase 3 \
	--num_epochs 25 \
	--train_inner_iter 13 \
	--sim_inner_iter 8 \
	--data_root_dir $1/parallel_wf/data/ \
	--model_dir ${model_dir} \
       	--mat_size 3000 \
	--num_mult 768 \
	--num_allreduce 2 \
	--sim_rank 128 \
	--train_rank 4 \
	--train_preprocess_time 10 \
	--sim_read_size 85899345920 \
        --sim_write_size 21474836480 \
        --train_read_size 6442450944 \
	--project_id RECUP
