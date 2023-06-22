#!/bin/bash

model_dir=$1/Ex3/model/
mkdir -p ${model_dir}

python ../rct-scripts/new_serial_multiphase_rct_v2.py \
	--num_phases 2 \
	--num_epochs 10 \
	--data_root_dir $1/Ex3/data/ \
	--model_dir ${model_dir} \
       	--mat_size 3000 \
	--num_mult 300 \
	--sim_rank 32 \
	--train_rank 4 \
	--project_id RECUP

