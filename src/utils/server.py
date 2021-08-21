
from typing import List, Dict

import torch
from torch import nn, Tensor
from torch.utils.data.dataloader import DataLoader
from torch.utils.data import Dataset
from torchvision.transforms import ToTensor
from torchvision import datasets
import torchaudio


from utils.audio import SubsetSC, collate_fn, number_of_correct, get_likely_index, set_LABELS, count_parameters
from utils.client import Client
from utils.model import FashionMNIST_CNN, SpeechCommand_M5



class Server():
    def __init__(self,
            task: str,
            clients: List[Client],
            epoch_num: int,
            device: str="cpu",
            ):

        self.task = task
        self.clients = clients
        self.epoch_num = epoch_num
        self.device = device
        # set in init_task
        self.model: nn.Module = None
        self.test_dataloader: DataLoader = None
        self.transform: nn.Module = None

    def init_task(self) -> nn.Module:
        if self.task == "FashionMNIST":
            test_dataset = datasets.FashionMNIST(
                root="~/projects/fledge/data/",
                train=False,
                download=True,
                transform=ToTensor()
                )
            self.test_dataloader = DataLoader(test_dataset, batch_size=64, drop_last=True)
            self.model = FashionMNIST_CNN()
        elif self.task == "SpeechCommand":
            test_dataset = SubsetSC("testing")
            if self.device == "cuda":
                num_workers = 1
                pin_memory = True
            else:
                num_workers = 0
                pin_memory = False
            self.test_dataloader = DataLoader(
                test_dataset,
                batch_size=64,
                shuffle=False,
                drop_last=True,
                collate_fn=collate_fn,
                num_workers=num_workers,
                pin_memory=pin_memory,
                )

            waveform, sample_rate, label, speaker_id, utterance_number = test_dataset[0]
            labels = sorted(list(set(datapoint[2] for datapoint in test_dataset)))
            set_LABELS(labels)
            new_sample_rate = 8000
            self.transform = torchaudio.transforms.Resample(orig_freq=sample_rate, new_freq=new_sample_rate)
            # transformed = self.transform(waveform)
            self.model = SpeechCommand_M5(
                # n_input=transformed.shape[0],
                # n_output=len(labels)
                )

    def distribute_model(self):
        """
        Send global model to clients.
        """
        for client in self.clients:
            client.model = self.model
            

    def aggregate_model(self):
        # collect models from clients
        state_dict_list = [client.model.state_dict() for client in self.clients]

        # calculate average model
        state_dict_avg = state_dict_list[0]
        for key in state_dict_avg.keys():
            for i in range(1, len(state_dict_list)):
                state_dict_avg[key] += state_dict_list[i][key]
            state_dict_avg[key] = torch.div(state_dict_avg[key], len(state_dict_list))
        
        # load average model
        self.model.load_state_dict(state_dict_avg)
        self.model = self.model.to(self.device)
        # self.model_state_dict = self.model.state_dict()
        # print("model len after aggregation: %d" % len(pickle.dumps(self.model.state_dict())))

    def test_model(self) -> float:
        self.model.eval()
        correct = 0
        for data, target in self.test_dataloader:
            data = data.to(self.device)
            target = target.to(self.device)
            # apply transform and model on whole batch directly on device
            data = self.transform(data)
            output = self.model(data)

            pred = get_likely_index(output)
            # pred = output.argmax(dim=-1)
            correct += number_of_correct(pred, target)
        # print("Accuracy: %.1f%\n" % 100. * correct / len(self.test_dataloader.dataset))
        return 1.0 * correct / len(self.test_dataloader.dataset)
