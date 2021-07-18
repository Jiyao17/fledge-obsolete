
run_num=$1
epoch_num=$2
lr_base=$3
data_per_client=$4
task=$5
client_nums=( 3 6 9 12 )
result_file=result.txt


source ~/fledge/python/bin/activate

for ((i=0; i<${#client_nums[@]}; i++))
do
    echo "run test for $run_num times"
    lr=`echo $lr_base ${client_nums[$i]} | awk '{printf "%1.2f\n", $1*$2}'` 
    # lr=$lr_base
    echo "local learning rate: $lr"
    echo "client num: ${client_nums[$i]}"
    echo "data per client: $data_per_client"
    echo "epoch num: $epoch_num"
    echo "task: $task"

    echo "lr=$lr client_num=${client_nums[$i]} data per client=$data_per_client \
        epoch num=$epoch_num task=$task" >> $result_file

    for ((j=0; j<$run_num; j++))
    do
        python ./server/main.py ${client_nums[$i]} $epoch_num $task &
        server_pid=$!

        sleep 10

        python ./client/main.py $lr ${client_nums[$i]} $data_per_client $epoch_num $task &
        clients_pid=$!

        wait $server_pid
        wait $clients_pid
    done
done

