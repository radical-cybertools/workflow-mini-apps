#!/bin/bash

exp_dir=/tmp/miniapp-exp/ddmd

pd=$1; shift

if test -z "$pd"
then
    echo "Error! Pilot description file not provided"
    exit -1
fi

if [ -d "$exp_dir" ]
then
    printf 'Directory '$exp_dir' exists - remove? (Y/n)? '
    read rep

    if test -z "$rep" -o "$rep" = "Y" -o "$rep" = "y"
    then
        rm -rf $exp_dir
    else
        echo "Error! Directory '$exp_dir' exists"
        exit -1
    fi
fi

mkdir -p "$exp_dir/model"
mkdir -p "$exp_dir/data"

echo "$(pwd)/io_size.json"

python ../rct-scripts/ddmd_async.py                   \
        --num_phases            3                     \
        --mat_size              10000                 \
        --data_root_dir         "$exp_dir/data"       \
        --num_step              60000                 \
        --num_epochs_train      150                   \
        --model_dir             "$exp_dir/model"      \
        --num_sample            500                   \
        --num_mult_train        4000                  \
        --dense_dim_in          12544                 \
        --dense_dim_out         128                   \
        --preprocess_time_train 30                    \
        --preprocess_time_agent 5                     \
        --num_epochs_agent      100                   \
        --num_mult_agent        1000                  \
        --num_mult_outlier      100                   \
        --num_sim               3                     \
        --work_dir              "$(pwd)"              \
        --io_json_file          "$(pwd)/io_size.json" \
        --pilot_description     "$(pwd)/$pd"

