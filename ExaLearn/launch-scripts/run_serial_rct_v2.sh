#!/bin/bash

model_dir=$1/serial_wf/model/
mkdir -p ${model_dir}

python ../rct-scripts/new_serial_multiphase_rct_v2.py \
	--num_phases 3 \
	--num_epochs 60 \
	--inner_iter 15 \
	--data_root_dir $1/serial_wf/data/ \
	--model_dir ${model_dir} \
       	--mat_size 2000 \
	--num_mult 300 \
	--sim_rank 32 \
	--train_rank 4 \
	--train_preprocess_time 15 \
	--sim_read_size 429496729 \
	--sim_write_size 107374182 \
	--train_read_size 53687091 \
	--project_id RECUP

