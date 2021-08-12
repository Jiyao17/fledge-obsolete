
run_num=$1
epoch_num=$2
local_epoch_num=$3
lr_base=$4
data_per_client=$5
task=$6
client_nums=( 3 )
result_file="result.txt"


source ~/fledge/python/bin/activate

for ((i=0; i<${#client_nums[@]}; i++))
do
    echo "run test for $run_num times"
    # lr_local=`echo $lr_base ${client_nums[$i]} | awk '{printf "%1.2f\n", $1*$2}'` 
    lr_local=$lr_base
    echo "local learning rate: $lr_local"
    echo "client num: ${client_nums[$i]}"
    echo "data per client: $data_per_client"
    echo "global epoch num: $epoch_num"
    echo "local epoch num: $local_epoch_num"
    echo "task: $task"

    args="lr_local=$lr_local client_num=${client_nums[$i]} data_per_client=$data_per_client"
    args1="global_epoch_num=$epoch_num local_epoch_num=$local_epoch_num task=$task"
    echo "" >> $result_file
    echo "$args $args1" >> $result_file

    for ((j=0; j<$run_num; j++))
    do
        python ./server/main.py ${client_nums[$i]} $epoch_num $task &
        server_pid=$!

        sleep 10

        python ./client/main.py $lr_local ${client_nums[$i]} $data_per_client $epoch_num $local_epoch_num $task &
        clients_pid=$!

        wait $server_pid
        wait $clients_pid
        echo "" >> $result_file
    done

    echo "" >> $result_file

done
