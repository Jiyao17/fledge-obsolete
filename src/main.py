
from typing import List

import torch

from utils.server import Server
from utils.client import Client
from utils.funcs import get_argument_parser, get_datasets


def check_device(target_device: str, real_device: str):
    if target_device == "cuda" and target_device != real_device:
        print("Warning: inconsistence of target device (%s) \
            and real device equipped(%s)" %
            (target_device, real_device)
        )
        return False
    else:
        return True

def run_sim(server: Server):

    for i in range(server.epoch_num):
        server.distribute_model()

        l_accuracy: List[float]
        for j in range(len(clients)):
            clients[j].train_model()
            l_accuracy[j] = clients[j].test_model()
        
        server.aggregate_model()
        g_accuracy = server.test_model()

        print("Epoch %d ......" % i)
        print("Global accuracy:\n%.2f%" % g_accuracy*100)
        print("Local accuracy:")
        print(l_accuracy)

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

    # input check
    # task check
    SUPPORTED_TASKS = ["FashionMNIST", "SpeechCommand"]
    if TASK not in SUPPORTED_TASKS:
        raise "Task not supported."
    # check device
    real_device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if check_device(DEVICE, real_device) == False:
        raise "CUDA required by input settings but not equipped."

    # partition data
    datasets = get_datasets(TASK, CLIENT_NUM, L_DATA_NUM, L_BATCH_SIZE, DATA_PATH)

    # initialize server and clients
    clients: List[Client] = [
        Client(TASK, datasets[i], L_EPOCH_NUM, DEVICE) 
        for i in range(CLIENT_NUM)
        ]
    server = Server(TASK, clients,  G_EPOCH_NUM, DEVICE)

    print("Args: %s %d %d %d %d %d %f %s %s %s" %
        (TASK, G_EPOCH_NUM, CLIENT_NUM, L_DATA_NUM, L_EPOCH_NUM, L_BATCH_SIZE, L_LR, DATA_PATH, DEVICE, RESULT_FILE)
        )

    # run_sim(server)


