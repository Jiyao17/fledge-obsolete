
from typing import List, Dict
import copy

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
            test_dataset: Dataset,
            clients: List[Client],
            epoch_num: int,
            device: str="cpu",
            ):

        self.task = task
        self.test_dataset = test_dataset
        self.clients = clients
        self.epoch_num = epoch_num
        self.device = device
        # set in self.init_task()
        self.model: nn.Module = None
        self.test_dataloader: DataLoader = None
        self.transform: nn.Module = None
        self.state_dicts: List[Dict[str, Tensor]] = None
        self.init_task()

        self.test_list = [1]

    def init_task(self):
        if self.task == "FashionMNIST":
            self.test_dataloader = DataLoader(self.test_dataset, batch_size=64, drop_last=True)
            self.model = FashionMNIST_CNN()
        elif self.task == "SpeechCommand":
            if self.device == "cuda":
                num_workers = 1
                pin_memory = True
            else:
                num_workers = 0
                pin_memory = False
            self.test_dataloader = DataLoader(
                self.test_dataset,
                batch_size=64,
                shuffle=False,
                drop_last=True,
                collate_fn=collate_fn,
                num_workers=num_workers,
                pin_memory=pin_memory,
                )

            waveform, sample_rate, label, speaker_id, utterance_number = self.test_dataset[0]
            labels = sorted(list(set(datapoint[2] for datapoint in self.test_dataset)))
            set_LABELS(labels)
            new_sample_rate = 8000
            self.transform = torchaudio.transforms.Resample(orig_freq=sample_rate, new_freq=new_sample_rate)
            transformed = self.transform(waveform)
            self.transform = self.transform.to(self.device)
            self.model = SpeechCommand_M5(
                n_input=transformed.shape[0],
                n_output=len(labels)
                )
        self.state_dicts = [
            client.model.state_dict()
            for client in self.clients
            ]

    def distribute_model(self):
        """
        Send global model to clients.
        """
        state_dict = self.model.state_dict()
        for client in self.clients:
            new_state_dict = copy.deepcopy(state_dict)
            client.model.load_state_dict(new_state_dict)

    def aggregate_model(self):
        state_dicts = [
            client.model.state_dict()
            for client in self.clients
            ]
        # calculate average model
        state_dict_avg = state_dicts[0]
        for key in state_dict_avg.keys():
            for i in range(1, len(state_dicts)):
                state_dict_avg[key] += state_dicts[i][key]
            state_dict_avg[key] = torch.div(state_dict_avg[key], len(self.state_dicts))
        
        self.model.load_state_dict(state_dict_avg)

    def test_model(self) -> float:
        self.model = self.model.to(self.device)
        self.model.eval()
        if self.task == "FashionMNIST":
            return self._test_FashionMNIST()
        if self.task == "SpeechCommand":
            return self._test_SpeechCommand()

    def _test_FashionMNIST(self):
        size = len(self.test_dataloader.dataset)
        test_loss, correct = 0, 0

        # with torch.no_grad():
        for X, y in self.test_dataloader:
            pred = self.model(X.to(self.device))
            # test_loss += loss_fn(pred, y.to(self.device)).item()
            correct += (pred.argmax(1) == y.to(self.device)).type(torch.float).sum().item()
        correct /= size

        return correct

    def _test_SpeechCommand(self):
        dataset_size = len(self.test_dataloader.dataset)
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

        return 1.0 * correct / dataset_size


