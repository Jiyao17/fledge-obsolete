
# controler config
run_num=$1
client_nums=( 3 6 9 12 )
local_data_num=5000 # data per client

# single task config
task=$2
global_epoch_num=$3
local_data_num=$4
local_epoch_num=$5
local_batch_size=$6
local_lr=$7

data_path="./data"
device="cuda"
result_file="./result.txt"
verbosity=1

source ./python/bin/activate
for ((i=0; i<${#client_nums[@]}; i++))
do
    CUDA_VISIBLE_DEVICES=2 python ./main.py $task $global_epoch_num ${client_nums[$i]} \
        $local_data_num $local_epoch_num $local_batch_size $local_lr \
        -p $data_path -d $device -r $result_file -v $verbosity -r $run_num

done
