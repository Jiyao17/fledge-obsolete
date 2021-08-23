
from io import TextIOWrapper
from multiprocessing.context import Process
from multiprocessing import Process, Queue, set_start_method
from typing import List

import torch
from utils.server import Server
from utils.client import Client
from utils.funcs import get_argument_parser, check_device, get_partitioned_datasets, get_test_dataset


def run_sim(que: Queue, progress_file: str, task, g_epoch_num, client_num, l_data_num, l_epoch_num, l_batch_size, l_lr, data_path, device, verbosity):
    # partition data
    datasets = get_partitioned_datasets(task, client_num, l_data_num, l_batch_size, data_path)
    test_dataset = get_test_dataset(task, data_path)
    # initialize server and clients
    clients: List[Client] = [
        Client(task, datasets[i], l_epoch_num, l_batch_size, l_lr, device) 
        for i in range(client_num)
        ]
    server = Server(task, test_dataset, clients, g_epoch_num, device)

    result: List[float] = []
    pf = open(progress_file, "a")
    for i in range(server.epoch_num):
        if verbosity >= 1:
            print("Epoch %d ......" % i)

        server.distribute_model()
        for j in range(len(server.clients)):
            server.clients[j].train_model()
        server.aggregate_model()
        # l_accuracy = [client.test_model() for client in server.clients]
        g_accuracy = server.test_model()
        if verbosity >= 1:
            print(f"Global accuracy:{g_accuracy*100:.2f}%")
            pf.write(f"Epoch {i}: {g_accuracy*100:.2f}%\n")
            if i % 10 == 9:
                pf.flush()
            # print(f"Local accuracy after training: {[acc for acc in l_accuracy]}")
        if i % 10 == 9:
            result.append(g_accuracy)

    que.put(result)


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
    RUN_NUM: int = args.run_num
    PROGRESS_FILE: str = args.progress_file

    if VERBOSITY >= 2:
        print("Input args: %s %d %d %d %d %d %f %s %s %s" %
            (TASK, G_EPOCH_NUM, CLIENT_NUM, L_DATA_NUM, L_EPOCH_NUM, L_BATCH_SIZE, L_LR, DATA_PATH, DEVICE, RESULT_FILE)
            )

    # input check
    SUPPORTED_TASKS = ["FashionMNIST", "SpeechCommand"]
    if TASK not in SUPPORTED_TASKS:
        raise "Task not supported!"
    if check_device(DEVICE) == False:
        raise "CUDA required by input but not equipped!"

    # run_sim(Queue(), TASK, G_EPOCH_NUM, CLIENT_NUM, L_DATA_NUM, L_EPOCH_NUM, L_BATCH_SIZE, L_LR, DATA_PATH, DEVICE, RESULT_FILE, VERBOSITY)
    # exit()

    set_start_method("spawn")
    que = Queue()
    procs: List[Process] = []

    for i in range(RUN_NUM):
        proc = Process(
                target=run_sim,
                args=(que, PROGRESS_FILE, TASK, G_EPOCH_NUM, CLIENT_NUM, L_DATA_NUM, L_EPOCH_NUM, L_BATCH_SIZE, L_LR, DATA_PATH, DEVICE, VERBOSITY)
            )
        proc.start()
        procs.append(proc)

    for proc in procs:
        proc.join()

    with open(RESULT_FILE, "a") as f:
        args = "{:12} {:11} {:10} {:10} {:11} {:12} {:4}".format(
            TASK, G_EPOCH_NUM, CLIENT_NUM, L_DATA_NUM, L_EPOCH_NUM, L_BATCH_SIZE, L_LR
            )
        f.write(
            "TASK          G_EPOCH_NUM CLIENT_NUM L_DATA_NUM L_EPOCH_NUM L_BATCH_SIZE L_LR\n" +
            args + "\n"
            )

        while que.empty() == False:
            result = que.get()
            print(result)
            [f.write(f"{num*100:.2f}% ") for num in result]
            f.write("\n")
        
        f.write("\n")