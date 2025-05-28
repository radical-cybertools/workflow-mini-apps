#!/bin/bash

exp_dir=/lustre/orion/scratch/tianle/lgt104/test_miniapp_ddmd_v1/

if [ -d ${exp_dir} ]
then
	echo "Error! Directory ${exp_dir} exists"
	exit -1
fi

mkdir -p ${exp_dir}/model
mkdir -p ${exp_dir}/data


python ../rct-scripts/ddmd-serial.py	\
    --config config_frontier.json \
    --num_phases		3		\
    --mat_size 		10000		\
    --data_root_dir		"${exp_dir}/data"	\
    --num_step		60000		\
    --num_epochs_train	150		\
    --model_dir		"${exp_dir}/model"	\
    --num_sample		500		\
    --num_mult_train	4000		\
    --dense_dim_in		12544		\
    --dense_dim_out		128		\
    --preprocess_time_train	30		\
    --preprocess_time_agent	5		\
    --num_epochs_agent	100		\
    --num_mult_agent	1000		\
    --num_mult_outlier	100		\
    --project_id		PHY157-ecpdwf		\
    --queue			"batch"		\
    --num_sim		12		\
    --num_nodes		2		\
    --io_json_file		"io_size-frontier.json"
