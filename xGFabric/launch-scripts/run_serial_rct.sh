#!/bin/bash

model_dir=$1/Ex3/model/
mkdir -p ${model_dir}

python ../rct-scripts/new_serial_multiphase_rct_v1.py \
	--num_phase 4 \
	--data_root_dir $1/Ex3/data/ \
	--model_dir ${model_dir} \
	--exec_pattern multi-thread \
       	--mat_size 900 \
	--project_id <project-id>
