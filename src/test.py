
from typing import List

from utils.client import Client
from utils.funcs import get_partitioned_datasets, get_test_dataset


task = "AG_NEWS"
client_num = 1
l_data_num = 10000
l_batch_size = 64
l_epoch_num = 5
l_lr = 0.01
device = "cuda"
data_path = "/home/tuo28237/projects/fledge/data/"

datasets = get_partitioned_datasets(task, client_num, l_data_num, l_batch_size, data_path)
test_dataset = get_test_dataset(task, data_path)
# initialize server and clients
# train_iter = AG_NEWS(split="train")
# train_dataset = to_map_style_dataset(train_iter)
clients: List[Client] = [
    Client(task, datasets[i], l_epoch_num, l_batch_size, l_lr, device) 
    for i in range(client_num)
    ]

for i in range(100):
    print("Epoch %d ......" % i)

    # server.distribute_model()
    for j in range(len(clients)):
        # acc = clients[j].test_model()
        # print("before training: client %d accuracy %.9f at epoch %d" % (j, acc, i))
        acc = clients[j].train_model()
        # print("training acc:  client %d accuracy %.9f at epoch %d" % (j, acc, i))
        acc = clients[j].test_model()
        print("after training:  client %d accuracy %.9f at epoch %d" % (j, acc, i))

