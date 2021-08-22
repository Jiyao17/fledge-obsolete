
# controler config
run_num=$1
client_nums=( 3 6 9 12 )
local_data_nums=( 2000 5000 10000 20000 60000 )

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
    echo "run test $run_num times for each config"
    # lr_local=`echo $lr_base ${client_nums[$i]} | awk '{printf "%1.2f\n", $1*$2}'` 
    echo "task: $task"
    echo "global epoch num: $epoch_num"
    echo "client num: ${client_nums[$i]}"
    echo "data per client: $data_per_client"
    echo "local epoch num: $local_epoch_num"
    echo "local batch size: $local_batch_size"
    echo "local learning rate: $local_lr"

    args="task=$task, global_epoch_num=$global_epoch_num, client_num=${client_nums[$i]}, \
    data per client: $data_per_client, local_epoch_num=$local_epoch_num, \
    local batch size: $local_batch_size, local learning rate=$local_lr"
    echo "" >> $result_file
    echo "$args" >> $result_file

    for ((j=0; j<$run_num; j++))
    do
        CUDA_VISIBLE_DEVICES=2 python ./main.py $task $epoch_num ${client_nums[$i]} \
            $epoch_num $task &
        server_pid=$!

        sleep 5

        # CUDA_VISIBLE_DEVICES=2 python ./client/main.py $lr_local ${client_nums[$i]} $data_per_client $epoch_num $local_epoch_num $task &
        # clients_pid=$!

        # wait $server_pid
        # wait $clients_pid
        echo "" >> $result_file
    done

    echo "" >> $result_file

done
