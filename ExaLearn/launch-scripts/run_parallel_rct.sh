#!/bin/bash

model_dir=$1/Ex4/model/
mkdir -p ${model_dir}

python ../rct-scripts/new_parallel_multiphase_rct_v1.py \
	--num_phase 4 \
	--data_root_dir $1/Ex4/data/ \
	--model_dir ${model_dir} \
	--exec_pattern multi-thread \
       	--mat_size 900 \
	--project_id CSC249ADCD08 
