
from typing import List, Dict
import copy

import torch
from torch import nn, Tensor
from torch.utils.data.dataloader import DataLoader
from torch.utils.data import Dataset
import torchaudio

from utils.audio import collate_fn, number_of_correct, get_likely_index, set_LABELS, count_parameters
from utils.client import Client
from utils.model import FashionMNIST_CNN, SpeechCommand_M5, AG_NEWS_TEXT
# from utils.text import collate_batch, vocab_size, emsize, num_class

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
        # used for repeat tests by self.reset_model()
        self.init_model = None
        self.init_task()

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
        elif self.task == "AG_NEWS":
            self.model = AG_NEWS_TEXT()
            self.test_dataloader = DataLoader(
                self.test_dataset,
                batch_size=64,
                shuffle=False,
                collate_fn=self.model.collate_batch)
            self.model = AG_NEWS_TEXT()


        # self.state_dicts = [
        #     client.model.state_dict()
        #     for client in self.clients
        #     ]

        # dic = self.model.state_dict()
        # self.init_model_dict = copy.deepcopy(dic)
        # del(dic)

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
        state_dict_avg = copy.deepcopy(state_dicts[0]) 
        for key in state_dict_avg.keys():
            for i in range(1, len(state_dicts)):
                state_dict_avg[key] += state_dicts[i][key]
            state_dict_avg[key] = torch.div(state_dict_avg[key], len(state_dicts))
        
        self.model.load_state_dict(state_dict_avg)

    def test_model(self) -> float:
        self.model = self.model.to(self.device)
        self.model.eval()
        if self.task == "FashionMNIST":
            return self._test_FashionMNIST()
        elif self.task == "SpeechCommand":
            return self._test_SpeechCommand()
        elif self.task == "AG_NEWS":
            return self._test_AG_NEWS()

    def reset_model(self):
        dic = copy.deepcopy(self.init_model_dict)
        self.model.load_state_dict(dic)
        del(dic)

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

    def _test_AG_NEWS(self):
        total_acc, total_count = 0, 0
        
        for idx, (label, text, offsets) in enumerate(self.test_dataloader):
            predicted_label = self.model(text, offsets)
            # loss = self.loss_fn(predicted_label, label)
            total_acc += (predicted_label.argmax(1) == label).sum().item()
            total_count += label.size(0)
            
        return total_acc/total_count



