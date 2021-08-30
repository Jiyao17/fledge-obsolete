
from typing import List
from argparse import ArgumentParser

import torch
from torchvision import datasets
from torchvision.transforms import ToTensor
from torch.utils.data import DataLoader, Subset, random_split

from utils.audio import SubsetSC


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
    ap.add_argument("-p", "--datapath", type=str, default="/home/tuo28237/projects/fledge/data/")
    ap.add_argument("-d", "--device", type=str, default="cpu")
    ap.add_argument("-r", "--result_file", type=str, default="./result.txt")
    ap.add_argument("-v", "--verbosity", type=int,default=1)
    ap.add_argument("-n", "--run_num", type=int, default=1)
    ap.add_argument("-f", "--progress_file", type=str, default="./progress.txt")

    return ap

def check_device(target_device: str) -> bool:
    real_device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if target_device != real_device and target_device == "cuda":
        print("Error: inconsistence of target device (%s) " \
            "and real device equipped (%s)" %
            (target_device, real_device)
        )
        return False
    else:
        return True

def get_partitioned_datasets(
    task: str,
    client_num: int,
    data_num: int,
    batch_size: int,
    data_path: str) \
    -> List[Subset]:

    if task == "FashionMNIST":
        train_dataset = datasets.FashionMNIST(
            root=data_path,
            train=True,
            download=True,
            transform=ToTensor(),
            )
    elif task == "SpeechCommand":
        train_dataset = SubsetSC("training", data_path)
    elif task == "AG_NEWS":
        pass

    dataset_size = len(train_dataset)
    # subset division
    if data_num * client_num > dataset_size:
        raise "No enough data!"
    data_num_total = data_num*client_num
    subset = random_split(train_dataset, [data_num_total, len(train_dataset)-data_num_total])[0]
    subset_lens = [ data_num for j in range(client_num) ]
    subsets = random_split(subset, subset_lens)
    
    return subsets

def get_test_dataset(task: str, data_path: str) -> Subset:
    if task == "FashionMNIST":
        test_dataset = datasets.FashionMNIST(
            root=data_path,
            train=False,
            download=True,
            transform=ToTensor()
            )
    elif task == "SpeechCommand":
        test_dataset = SubsetSC("testing", data_path)
    elif task == "AG_NEWS":
        pass

    return test_dataset
