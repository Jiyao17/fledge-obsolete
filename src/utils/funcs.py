

from typing import List
from argparse import ArgumentParser

from torchvision import datasets
from torchvision.transforms import ToTensor
from torch.utils.data import DataLoader
from torch.utils.data import DataLoader, Subset, dataset, Dataset, random_split

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
    ap.add_argument("-p", "--datapath", type=str, default="../data/")
    ap.add_argument("-d", "--device", type=str, default="cpu")
    ap.add_argument("-r", "--result_file", type=str, default="./result.txt")

    return ap

def get_datasets(
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

    dataset_size = len(train_dataset)
    # subset division
    if data_num * client_num > dataset_size:
        raise "No enough data!"
    data_num_total = data_num*client_num
    subset = random_split(train_dataset, [data_num_total, len(train_dataset)-data_num_total])[0]
    subset_lens = [ data_num for j in range(client_num) ]
    subsets = random_split(subset, subset_lens)
    
    return subsets


