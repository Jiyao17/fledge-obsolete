
from typing import List

import torch

from utils.server import Server
from utils.client import Client
from utils.funcs import get_argument_parser, check_device, get_partitioned_datasets, get_test_dataset

def run_sim(server: Server, verbosity: int = 1):

    for i in range(server.epoch_num):
        server.distribute_model()

        l_accuracy: List[float]
        for j in range(len(clients)):
            clients[j].train_model()
            # l_accuracy[j] = clients[j].test_model()
        
        server.aggregate_model()
        g_accuracy = server.test_model()

        if VERBOSITY >= 1:
            print("Epoch %d ......" % i)
            print(f"Global accuracy: {g_accuracy*100:.2f}%")
            # print("Local accuracy:")
            # print(l_accuracy)

if __name__ == "__main__":

    ap = get_argument_parser()
    args = ap.parse_args()

    TASK: str = args.task # limited: FashionMNIST/SpeechCommand/
    # global parameters
    G_EPOCH_NUM: int = args.g_epoch_num
    # local parameters
    CLIENT_NUM: int = args.client_num
    L_DATA_NUM: int = args.l_data_num
    L_EPOCH_NUM: int = args.l_epoch_num
    L_BATCH_SIZE: int = args.l_batch_size
    L_LR: float = args.l_lr
    # shared settings
    DATA_PATH: str = args.datapath
    DEVICE: str = torch.device(args.device)
    RESULT_FILE: str = args.result_file
    VERBOSITY: int = args.verbosity

    if VERBOSITY >= 2:
        print("Input args: %s %d %d %d %d %d %f %s %s %s" %
            (TASK, G_EPOCH_NUM, CLIENT_NUM, L_DATA_NUM, L_EPOCH_NUM, L_BATCH_SIZE, L_LR, DATA_PATH, DEVICE, RESULT_FILE)
            )


    # input check
    SUPPORTED_TASKS = ["FashionMNIST", "SpeechCommand"]
    if TASK not in SUPPORTED_TASKS:
        raise "Task not supported."
    if check_device(DEVICE) == False:
        raise "Targeted and equipped devices inconsist."

    # partition data
    datasets = get_partitioned_datasets(TASK, CLIENT_NUM, L_DATA_NUM, L_BATCH_SIZE, DATA_PATH)
    test_dataset = get_test_dataset(TASK, DATA_PATH)
    # initialize server and clients
    clients: List[Client] = [
        Client(TASK, datasets[i], L_EPOCH_NUM, L_BATCH_SIZE, L_LR, DEVICE) 
        for i in range(CLIENT_NUM)
        ]
    server = Server(TASK, test_dataset, clients, G_EPOCH_NUM, DEVICE)

    run_sim(server, VERBOSITY)

    

