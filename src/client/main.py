

from typing import List
import sys
import pickle

import torch
from torch import optim, nn
from torch.nn.modules.loss import CrossEntropyLoss
from torch.utils.data import DataLoader, Subset, dataset, Dataset, random_split
from torchvision import datasets
from torchvision.transforms import ToTensor
import torchaudio

import torch.nn.functional as F

from utils.client import Client
from utils.model import FashionMNIST_CNN, SpeechCommand_M5
from utils.audio import SubsetSC, collate_fn


batch_size=32

if __name__ == "__main__":
    
    # read arguments
    lr: float = float(sys.argv[1])     # learning rate
    client_num: int = int(sys.argv[2])
    data_num_per_client: int = int(sys.argv[3])
    g_epoch_num: int = int(sys.argv[4])
    local_epoch_num: int = int(sys.argv[5])
    task: str = sys.argv[6] # limited: FashionMNIST/SpeechCommand/
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
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

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
    elif task == "SpeechCommand":
        train_dataset = SubsetSC("training")
        waveform, sample_rate, label, speaker_id, utterance_number = train_dataset[0]
        labels = sorted(list(set(datapoint[2] for datapoint in train_dataset)))
        new_sample_rate = 8000
        transform = torchaudio.transforms.Resample(orig_freq=sample_rate, new_freq=new_sample_rate)
        transformed = transform(waveform)

        if device == "cuda":
            num_workers = 1
            pin_memory = True
        else:
            num_workers = 0
            pin_memory = False
    
        model = SpeechCommand_M5(
            n_input=transformed.shape[0],
            n_output=len(labels),
            transform=transform
            )

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

    # clients initialization
    for i in range(client_num):
        print("Loading data......")
        # data_range = [j for j in range(dataset_size//client_num *i, dataset_size//client_num *(i+1))]
        print("dataset size per client: %d" % len(subset_list[i]))
        
        if task == "FashionMNIST":
            dataloader = DataLoader(subset_list[i], batch_size=batch_size, shuffle=True, drop_last=True)
            loss_fn = CrossEntropyLoss()
            optimizer = optim.SGD(model.parameters(), lr=lr)
            scheduler = None
        elif task == "SpeechCommand":
            dataloader = DataLoader(
                subset_list[i],
                batch_size=batch_size,
                shuffle=True,
                collate_fn=collate_fn,
                num_workers=num_workers,
                pin_memory=pin_memory,
                drop_last=True
                )

            loss_fn = F.nll_loss
            optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=0.0001)
            scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=20, gamma=0.1)  # reduce the learning after 20 epochs by a factor of 10

        model_len = len(pickle.dumps(model.state_dict()))
        print("raw model len: %d" % model_len)

        client = Client(
            task=task,
            dataloader=dataloader,
            model=model,
            loss_fn=loss_fn,
            optimizer=optimizer,
            scheduler=scheduler,
            epoch_num=local_epoch_num,
            device=device
            )
        client.init()
        client_list.append(client)

    # psuedo parallel training
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


    