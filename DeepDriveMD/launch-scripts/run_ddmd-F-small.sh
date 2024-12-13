#!/bin/bash

exp_dir=/eagle/RECUP/twang/miniapp-exp/ddmd-reprod-v2

if [ -d ${exp_dir} ]
then
	echo "Error! Directory ${exp_dir} exists"
	exit -1
fi

mkdir -p ${exp_dir}/model
mkdir -p ${exp_dir}/data

num_phase=2
for((i=0; i<num_phase; i++))
do
	mkdir -p ${exp_dir}/data/phase${i}
done


python ../rct-scripts/ddmd-F.py	                \
	--num_phases		    ${num_phase}	    \
	--mat_size 		        1000		        \
	--data_root_dir		    "${exp_dir}/data"	\
	--num_step		        4800		        \
	--num_epochs_train	    10		            \
	--model_dir		        "${exp_dir}/model"	\
	--conda_env		        "/eagle/RECUP/twang/env/rose-task-base-clone"	\
	--num_sample		    500		            \
	--num_mult_train	    400		            \
	--dense_dim_in		    12544	            \
	--dense_dim_out		    128		            \
	--preprocess_time_train	3		            \
	--preprocess_time_agent	5		            \
	--num_epochs_agent	    10		            \
	--num_mult_agent	    100		            \
	--num_mult_outlier	    10		            \
	--project_id		    RECUP		        \
	--queue			        "debug"	        	\
	--num_sim		        12		            \
	--num_nodes		        1		            \
	--io_json_file		    "io_size.json"  
#	--enable_darshan			\
