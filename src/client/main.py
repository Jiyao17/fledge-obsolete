

from typing import List
import sys
import pickle

from torch import optim, nn
from torch.nn.modules.loss import CrossEntropyLoss
from torch.utils.data import DataLoader, Subset, dataset, Dataset, random_split
from torchvision import datasets
from torchvision.transforms import ToTensor

from utils.client import Client
from utils.model import FashionMNIST_CNN


batch_size=32

if __name__ == "__main__":
    
    # read arguments
    lr: float = float(sys.argv[1])     # learning rate
    client_num: int = int(sys.argv[2])
    data_num_per_client: int = int(sys.argv[3])
    g_epoch_num: int = int(sys.argv[4])
    local_epoch_num: int = int(sys.argv[5])
    task = sys.argv[6]
    # batch_size = int(sys.argv[5])
    # data_path = sys.argv[6]
    # server_ip = sys.argv[7]
    # server_port = int(sys.argv[8])
    # device = sys.argv[9]    # training device

    # lr = 0.01     # learning rate
    # client_num = 6
    # data_num_per_client = 5000
    # g_epoch_num = 50
    batch_size = 64
    data_path = "~/fledge/data"
    server_ip = "127.0.0.1"
    server_port = 5000
    device = "cuda"    # training device

    train_dataset: Dataset = None
    model: nn.Module = None
    if task == "FashionMNIST":
        train_dataset = datasets.FashionMNIST(
            root=data_path,
            train=True,
            download=True,
            transform=ToTensor(),
            )
        model = FashionMNIST_CNN()
    elif task == "CIFAR":
        pass
    else:
        raise "Task not supported yet."


    dataset_size = len(train_dataset)
    client_list: List[Client] = []

    # subset division
    if data_num_per_client * client_num > dataset_size:
        raise "No enough data!"
    data_num = data_num_per_client*client_num
    subset = random_split(train_dataset, [data_num, len(train_dataset)-data_num])[0]

    subset_lens = [ data_num_per_client for j in range(client_num) ]
    subset_list = random_split(subset, subset_lens)

    for i in range(client_num):
        print("Loading data......")
        # data_range = [j for j in range(dataset_size//client_num *i, dataset_size//client_num *(i+1))]
        print("dataset size per client: %d" % len(subset_list[i]))
        dataloader = DataLoader(subset_list[i], batch_size=batch_size, shuffle=True, drop_last=True)
        

        model_len = len(pickle.dumps(model.state_dict()))
        print("raw model len: %d" % model_len)
        loss_fn = CrossEntropyLoss()
        optimizer = optim.SGD(model.parameters(), lr=lr)

        client = Client(dataloader=dataloader, model=model, optimizer=optimizer, device="cuda")
        client.init()
        client_list.append(client)


    for i in range(g_epoch_num):
        print("Epoch %d" % i)
        
        # print("Downloading model......")
        for j in range(client_num):
            # get model
            # print("Client %d downloading model......" % j)
            client_list[j].download_model()

        # print("Training new models......")
        for j in range(client_num):
            # print("Client %d training new model......" % j)
            client_list[j].train_model()

        # print("Uploading new models......")
        for j in range(client_num):
            # print("Client %d uploading new models......" % j)
            client_list[j].upload_model()


    