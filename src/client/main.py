
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
lr=0.01

dataset_size = 60000

client_num=3




if __name__ == "__main__":
    # device = "cuda" if torch.cuda.is_available() else "cpu"

    train_dataset = datasets.FashionMNIST(
        root="../data",
        train=True,
        download=True,
        transform=ToTensor()
        )

    config_file = "./config.json"
    client_list: List[Client] = []

    for i in range(client_num):
        # print("Loading data......")

        data_range = [j for j in range(dataset_size//client_num *i, dataset_size//client_num *(i+1))]
        dataloader = DataLoader(Subset(train_dataset, data_range), batch_size=batch_size)
        model = FashionMNIST_CNN()
        loss_fn = CrossEntropyLoss()
        optimizer = optim.SGD(model.parameters(), lr=lr)

        client = Client(config_file, dataloader, model, loss_fn, optimizer)
        client.init()
        client_list.append(client)

        # net_list.append(ClientNet(SERVER_ADDR, SERVER_PORT))
        # model_list.append(FashionMNIST_CNN())
        # data_range = [j for j in range(20000*i, 20000*(i+1))]
        # dataloader = DataLoader(Subset(train_dataset, data_range), batch_size=batch_size)

        # dataloader_list.append(dataloader)

    for i in range(local_epoch_num):
        print("Epoch %d" % i)
        
        print("Downloading model......")
        for j in range(client_num):
            # get model
            print("Client %d downloading model......" % j)
            client_list[j].download_model()

            # sd_len = net_list[j].recv(4)
            # print(int.from_bytes(sd_len, 'big'))
            # state_bytes = net_list[j].recv(int.from_bytes(sd_len, 'big'))
            # print("Downloaded model %d" % j)
            # state_dict = pickle.loads(state_bytes)
            # print("Loading model to GPU")
            # model_list[j].load_state_dict(state_dict)
            # model_list[j].to(device)

        print("Training new models......")
        for j in range(client_num):
            print("Client %d training new model......" % j)
            client_list[j].train_model()
            # train

            # print(len(dataloader_list[j]))
            # for batch, (X, y) in enumerate(dataloader_list[j]):
            #     # Compute prediction and loss
            #     pred = model_list[j](X.cuda())
            #     loss = loss_fn(pred, y.cuda())
                
            #     # Backpropagation
            #     optimizer_list[j].zero_grad()
            #     loss.backward()
            #     optimizer_list[j].step()

        print("Uploading new models......")
        for j in range(client_num):
            print("Client %d uploading new models......" % j)
            client_list[j].upload_model()
            # upload model
            # state_dict = model_list[j].state_dict()
            # state_bytes = pickle.dumps(state_dict)
            # net_list[j].send(len(state_bytes).to_bytes(4, 'big'))
            # net_list[j].send(state_bytes)


    