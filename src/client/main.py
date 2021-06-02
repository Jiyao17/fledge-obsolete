
import pickle
import torch
from torch import nn
from torch.utils.data import DataLoader, Subset
from torchvision import datasets
from torchvision.transforms import ToTensor, Lambda

from utils.net import ClientNet
from utils.model import FashionMNIST_CNN

import json

SERVER_ADDR="127.0.0.1"
SERVER_PORT=5000
local_epoch_num=40
batch_size=32
lr=0.01

client_num=3




if __name__ == "__main__":
    device = "cuda" if torch.cuda.is_available() else "cpu"

    train_dataset = datasets.FashionMNIST(
        root="data",
        train=True,
        download=True,
        transform=ToTensor()
        )

    loss_fn = nn.CrossEntropyLoss()

    net_list = []
    model_list = []
    dataloader_list = []
    optimizer_list = []

    for i in range(client_num):
        print("Loading data......")
        net_list.append(ClientNet(SERVER_ADDR, SERVER_PORT))
        model_list.append(FashionMNIST_CNN())
        data_range = [j for j in range(20000*i, 20000*(i+1))]
        dataloader = DataLoader(Subset(train_dataset, data_range), batch_size=batch_size)

        dataloader_list.append(dataloader)
        optimizer = torch.optim.SGD(model_list[i].parameters(), lr=lr)
        optimizer_list.append(optimizer)

    for i in range(local_epoch_num):
        
        print("Downloading models......")
        for j in range(client_num):
            # get model
            print("Downloading model %d......" % j)
            sd_len = net_list[j].recv(4)
            print(int.from_bytes(sd_len, 'big'))
            state_bytes = net_list[j].recv(int.from_bytes(sd_len, 'big'))
            print("Downloaded model %d" % j)
            state_dict = pickle.loads(state_bytes)
            print("Loading model to GPU")
            model_list[j].load_state_dict(state_dict)
            model_list[j].to(device)

        print("Training new models......")
        for j in range(client_num):
            # train

            print(len(dataloader_list[j]))
            for batch, (X, y) in enumerate(dataloader_list[j]):
                # Compute prediction and loss
                pred = model_list[j](X.cuda())
                loss = loss_fn(pred, y.cuda())
                
                # Backpropagation
                optimizer_list[j].zero_grad()
                loss.backward()
                optimizer_list[j].step()

        print("Uploading new models......")
        for j in range(client_num):
            # upload model
            state_dict = model_list[j].state_dict()
            state_bytes = pickle.dumps(state_dict)
            net_list[j].send(len(state_bytes).to_bytes(4, 'big'))
            net_list[j].send(state_bytes)


    