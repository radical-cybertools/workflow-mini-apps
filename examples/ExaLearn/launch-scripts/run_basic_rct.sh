#!/bin/bash

model_dir=$1/Ex2/model/
mkdir -p ${model_dir}

python ../rct-scripts/new_serial_multiphase_rct_v1.py \
	--num_phase 1 \
	--data_root_dir $1/Ex2/data/ \
	--model_dir ${model_dir} \
	--exec_pattern multi-thread \
       	--mat_size 1800 \
	--project_id CSC249ADCD08
