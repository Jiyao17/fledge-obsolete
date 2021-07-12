
import pickle
import torch
from typing import List


from torch import nn, optim
from torch.nn.modules.loss import CrossEntropyLoss
from torch.utils.data import DataLoader, Subset
from torchvision import datasets
from torchvision.transforms import ToTensor

from utils.client import Client
from utils.model import FashionMNIST_CNN


local_epoch_num=40
batch_size=32
lr=0.02

dataset_size = 60000

client_num=3




if __name__ == "__main__":
    # device = "cuda" if torch.cuda.is_available() else "cpu"

    train_dataset = datasets.FashionMNIST(
        root="~/fledge/data",
        train=True,
        download=True,
        transform=ToTensor()
        )

    config_file = "./config.json"
    client_list: List[Client] = []

    for i in range(client_num):
        print("Loading data......")

        data_range = [j for j in range(dataset_size//client_num *i, dataset_size//client_num *(i+1))]
        dataloader = DataLoader(Subset(train_dataset, data_range), batch_size=batch_size)
        model = FashionMNIST_CNN()
        loss_fn = CrossEntropyLoss()
        optimizer = optim.SGD(model.parameters(), lr=lr)

        client = Client(config_file, dataloader, model, loss_fn, optimizer)
        client.init()
        client_list.append(client)


    for i in range(local_epoch_num):
        print("Epoch %d" % i)
        
        print("Downloading model......")
        for j in range(client_num):
            # get model
            # print("Client %d downloading model......" % j)
            client_list[j].download_model()

        print("Training new models......")
        for j in range(client_num):
            # print("Client %d training new model......" % j)
            client_list[j].train_model()

        print("Uploading new models......")
        for j in range(client_num):
            # print("Client %d uploading new models......" % j)
            client_list[j].upload_model()


    