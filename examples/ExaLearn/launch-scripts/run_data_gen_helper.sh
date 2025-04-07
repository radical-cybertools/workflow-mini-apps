#!/bin/bash

exp_dir_name=$1
exp_idx=$2
num_phase=$3

if (($# != 3))
then
	echo "Number of argument is not 3!!!"
fi

python ../Executables/data_gen_helper.py \
	${exp_dir_name}/Ex${exp_idx}/configs/ \
	2.5 5.5 0.002 \
	2.5 5.5 0.01 \
	20 88 1 \
	2.5 5.5 0.01 \
	92 120 1 \
	3.5 4.5 0.02 \
	3.5 4.5 0.02 \
	${num_phase} \
	/home/twang3/myWork/miniapp-exalearn/RECUP/mini-apps/ExaLearn-final/cif_files/ \
	${exp_dir_name}/Ex${exp_idx}/data

# line 13: where to generate config file
# line 21: number of phases
# line 22: where are cif file
# line 23: where to output hdf5 data
