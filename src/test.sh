
run_num=$1
epoch_num=$2
lr_base=$3
data_per_client=$4
client_nums=( 3 6 9 12 )

source ~/fledge/python/bin/activate
echo "run test for $run_num times"

for ((i=0; i<${#client_nums[@]}; i++))
do
    for ((j=0; j<$run_num; j++))
    do
        python ./server/main.py ${client_nums[$i]} $epoch_num &
        server_pid=$!

        sleep 10

        lr=`echo $lr_base ${client_nums[$i]} | awk '{printf "%1.2f\n", $1*$2}'` 
        # lr=$lr_base
        echo "learning rate: $lr"
        python ./client/main.py $lr ${client_nums[$i]} $data_per_client $epoch_num &
        clients_pid=$!

        wait $server_pid
        wait $clients_pid
    done
done

