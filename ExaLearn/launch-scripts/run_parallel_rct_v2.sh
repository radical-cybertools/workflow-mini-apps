#!/bin/bash

model_dir=$1/parallel_wf/model/
mkdir -p ${model_dir}

python ../rct-scripts/new_parallel_multiphase_rct_v2.py \
	--num_phase 2 \
	--num_epochs 15 \
	--inner_iter 15 \
	--data_root_dir $1/parallel_wf/data/ \
	--model_dir ${model_dir} \
       	--mat_size 3000 \
	--num_mult 300 \
	--sim_rank 32 \
	--train_rank 4 \
	--train_preprocess_time 15 \
	--project_id RECUP
