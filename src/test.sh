#!/bin/bash

# controler config
progress_file="./progress.txt"
# program_stdout="program_stdout.txt" # stdout of the program, not the shell
local_data_num=5000 # data per client
client_nums=( 1 )

# single task config
task=$1
global_epoch_num=$2
local_data_num=$3
local_epoch_num=$4
local_batch_size=$5
local_lr=$6
run_num=$7

data_path="../data"
device="cuda"
result_file="./result.txt"
verbosity=2
        
source ../python/bin/activate
for ((i=0; i<${#client_nums[@]}; i++))
do
    echo " CUDA_VISIBLE_DEVICES=2 python ./main.py $task $global_epoch_num ${client_nums[$i]} \
        $local_data_num $local_epoch_num $local_batch_size $local_lr \
        -p $data_path -d $device -r $result_file -v $verbosity -n $run_num -f $progress_file"

#     echo "progress： client_num=${client_nums[$i]}" >> $progress_file
    CUDA_VISIBLE_DEVICES=2 python main.py $task $global_epoch_num ${client_nums[$i]} \
        $local_data_num $local_epoch_num $local_batch_size $local_lr \
        -p $data_path -d $device -r $result_file -v $verbosity -n $run_num -f $progress_file

done
