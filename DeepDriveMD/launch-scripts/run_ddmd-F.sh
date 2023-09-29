#!/bin/bash

exp_dir=/eagle/RECUP/twang/miniapp-exp/ddmd

if [ -d ${exp_dir} ]
then
	echo "Error! Directory ${exp_dir} exists"
	exit -1
fi

mkdir -p ${exp_dir}/model
mkdir -p ${exp_dir}/data

python ../rct-scripts/ddmd-F.py	\
	--num_phases		3		\
	--mat_size 		5000		\
	--data_root_dir		"${exp_dir}/data"	\
	--num_step		1000		\
	--num_epochs_train	150		\
	--model_dir		"${exp_dir}/model"	\
	--num_sample		500		\
	--num_mult_train	4000		\
	--dense_dim_in		12544		\
	--dense_dim_out		128		\
	--preprocess_time	20		\
	--num_epochs_agent	10		\
	--num_mult_agent	2000		\
	--num_mult_outlier	10		\
	--project_id		RECUP		\
	--num_sim		12		\
	--num_nodes		3		\
	--io_json_file		"io_size.json"
