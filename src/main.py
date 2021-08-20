
from typing import List
from argparse import ArgumentParser

import torch

from utils.server import Server
from utils.client import Client


def get_argument_parser() -> ArgumentParser:
    ap = ArgumentParser()
    # positional
    ap.add_argument("task", type=str)
    ap.add_argument("g_epoch_num", type=int)
    ap.add_argument("client_num", type=int)
    ap.add_argument("l_data_num", type=int)
    ap.add_argument("l_epoch_num", type=int)
    ap.add_argument("l_batch_size", type=int)
    ap.add_argument("l_lr", type=float)
    # optional
    ap.add_argument("-p", "--datapath", type=str, default="~/projects/fledge/data/")
    ap.add_argument("-d", "--device", type=str, default="cpu")
    ap.add_argument("-r", "--result_file", type=str, default="./result.txt")

    return ap

def run_sim(server: Server, clients: List[Client]):

    for i in range(server.epoch_num):
        server.distribute_model()

        l_accuracy: List[float]
        for j in range(len(clients)):
            clients[j].train_model()
            l_accuracy[j] = clients[j].test_model()
        
        server.aggregate_model()
        g_accuracy = server.test_model()

        print("Epoch %d" % i)
        print("Global accuracy:\n%.2f%" % g_accuracy*100)
        print("Local accuracy:")
        print(l_accuracy)

if __name__ == "__main__":

    ap: ArgumentParser = get_argument_parser()
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

    clients: List[Client] = [
        Client(TASK, L_EPOCH_NUM, DEVICE) 
        for i in range(CLIENT_NUM)
        ]
    server = Server(TASK, CLIENT_NUM, DEVICE, G_EPOCH_NUM)

    print("Args: %s %d %d %d %d %d %f %s %s %s" %
        (TASK, G_EPOCH_NUM, CLIENT_NUM, L_DATA_NUM, L_EPOCH_NUM, L_BATCH_SIZE, L_LR, DATA_PATH, DEVICE, RESULT_FILE)
        )

    # run_sim(server, clients)



    # if TASK == "FashoinMNIST":
    #     run_FashionMNIST(server, clients)
    # elif TASK == "SpeechCommand":
    #     run_SpeechCommand(server, clients)
    # else:
    #     raise "Unsupported task."

# def run_FashionMNIST(server: Server, clients: List[Client]):
#     pass

# def run_SpeechCommand(server: Server, clients: List[Client]):
#     pass