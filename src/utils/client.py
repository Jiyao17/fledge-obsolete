
import torch
from torch.optim.lr_scheduler import StepLR
from torch import nn, optim
from torch.optim import Optimizer
from torch.nn.modules import loss
from torch.utils.data import Dataset, DataLoader
from torchaudio.transforms import Resample
from torch.nn.modules.loss import CrossEntropyLoss
import torchaudio
import torch.nn.functional as F

import copy
from torchtext.datasets import AG_NEWS


from utils.model import FashionMNIST_CNN, SpeechCommand_M5, AG_NEWS_TEXT
from utils.audio import collate_fn, set_LABELS
from utils.funcs import get_test_dataset
# from utils.text import collate_batch, vocab_size, emsize, num_class
# from utils.text import collate_batch#, vocab_size, emsize, num_class



class Client():
    def __init__(self,
            task: str,
            train_dataset: Dataset,
            epoch_num: int=5,
            batch_size: int=256,
            lr: int=0.01,
            device: str="cpu"
            ):
        self.task = task
        self.train_dataset = train_dataset
        self.epoch_num = epoch_num
        self.batch_size=batch_size
        self.lr=lr
        self.device = device
        # set by self.init_task()
        self.train_dataloader: DataLoader = None
        self.transform: nn.Module = None
        self.loss_fn = None
        self.optimizer: Optimizer = None
        self.scheduler = None
        self.model: nn.Module = None
        self.init_task()

    def init_task(self) -> nn.Module:
        if self.task == "FashionMNIST":
            self._init_FashionMNIST()
        elif self.task == "SpeechCommand":
            self._init_SpeechCommand()
        elif self.task == "AG_NEWS":
            self._init_AG_NEWS()
        else:
            raise "Unsupported task."

    def train_model(self):
        self.model = self.model.to(self.device)
        self.model.train()
        for i in range(self.epoch_num):
            if self.task == "FashionMNIST":
                self._train_FashionMNIST()
            elif self.task == "SpeechCommand":
                self._train_SpeechCommand()
                self.scheduler.step()
            elif self.task == "AG_NEWS":
                acc = self._train_AG_NEWS_1()
                self.scheduler.step()

                return acc
            else:
                raise "Unsupported task."

    def _init_FashionMNIST(self):
        self.train_dataloader = DataLoader(
            self.train_dataset,
            batch_size=self.batch_size,
            shuffle=True,
            drop_last=True
            )
        self.loss_fn = CrossEntropyLoss()
        self.model = FashionMNIST_CNN()
        self.optimizer = optim.SGD(self.model.parameters(), lr=self.lr)
        self.scheduler = None

    def _init_SpeechCommand(self):
        if self.device == "cuda":
            num_workers = 1
            pin_memory = True
        else:
            num_workers = 0
            pin_memory = False

        self.dataloader = DataLoader(
            self.train_dataset,
            batch_size=self.batch_size,
            shuffle=True,
            collate_fn=collate_fn,
            num_workers=num_workers,
            pin_memory=pin_memory,
            drop_last=True
            )
        waveform, sample_rate, label, speaker_id, utterance_number = self.train_dataset[0]
        labels = sorted(list(set(datapoint[2] for datapoint in self.train_dataset)))
        set_LABELS(labels)
        new_sample_rate = 8000
        transform = torchaudio.transforms.Resample(orig_freq=sample_rate, new_freq=new_sample_rate)
        transformed: Resample = transform(waveform)
        self.transform = transform.to(self.device)
        self.loss_fn = F.nll_loss
        self.model = SpeechCommand_M5(
            n_input=transformed.shape[0],
            n_output=len(labels)
            )
        self.optimizer = optim.Adam(self.model.parameters(), lr=self.lr, weight_decay=0.0001)
        self.scheduler = optim.lr_scheduler.StepLR(self.optimizer, step_size=20, gamma=0.5)  # reduce the learning after 20 epochs by a factor of 10

    def _init_AG_NEWS(self):
        self.model = AG_NEWS_TEXT()
        # train_iter = AG_NEWS(split='train')
        self.train_dataloader = DataLoader(
            self.train_dataset,
            batch_size=self.batch_size,
            shuffle=True,
            collate_fn=self.model.collate_batch,
            drop_last=True)
        self.loss_fn = CrossEntropyLoss()
        self.optimizer = torch.optim.SGD(self.model.parameters(), lr=self.lr)
        self.scheduler = torch.optim.lr_scheduler.StepLR(self.optimizer, 1.0, gamma=0.1)

        self.test_dataset = get_test_dataset(self.task, "~/projects/fledge/data")
        self.test_dataloader = DataLoader(
            self.test_dataset,
            batch_size=64,
            shuffle=False,
            collate_fn=self.model.collate_batch,
            drop_last=True,
            )

    def _train_FashionMNIST(self):
        for batch, (X, y) in enumerate(self.train_dataloader):
            # Compute prediction and loss
            pred = self.model(X.cuda())
            loss = self.loss_fn(pred, y.cuda())
            # Backpropagation
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

    def _train_SpeechCommand(self):
        self.transform = self.transform.to(self.device)
        for batch_idx, (data, target) in enumerate(self.dataloader):
            data = data.to(self.device)
            target = target.to(self.device)
            # apply transform and model on whole batch directly on device
            data = self.transform(data)
            output = self.model(data)
            # negative log-likelihood for a tensor of size (batch x 1 x n_output)
            loss = self.loss_fn(output.squeeze(), target)

            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

    def _train_AG_NEWS(self):
        self.model.train()
        for idx, (label, text, offsets) in enumerate(self.train_dataloader, 0):
            
            predicted_label = self.model(text.cuda(), offsets.cuda())
            loss = self.loss_fn(predicted_label, label.cuda())
            self.optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), 0.1)
            self.optimizer.step()

    def _train_AG_NEWS_1(self):
        # self.m = copy.deepcopy(self.model)
        # self.m.train()
        total_acc, total_count = 0, 0

        for idx, (label, text, offsets) in enumerate(self.train_dataloader):
            self.optimizer.zero_grad()
            predicted_label = self.model(text, offsets)
            loss = self.loss_fn(predicted_label, label)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), 0.1)
            self.optimizer.step()

            
            total_acc += (predicted_label.argmax(1) == label).sum().item()
            total_count += label.size(0)
        
        # self.model = copy.deepcopy(self.m)
        return total_acc/total_count


    def test_model(self) -> float:
        # functionality of testing local model is not guaranteed yet
        self.model = self.model.to(self.device)
        self.model.eval()
        if self.task == "FashionMNIST":
            accuracy = self._test_FashionMNIST()
        if self.task == "SpeechCommand":
            accuracy = self._test_SpeechCommand()
        if self.task == "AG_NEWS":
            accuracy = self._test_AG_NEWS()
        return accuracy

    def _test_FashionMNIST(self):
        test_dataset = get_test_dataset("FashionMNIST", "~/projects/fledge/data")
        self.test_dataloader = DataLoader(test_dataset, batch_size=64, drop_last=True)
        size = len(self.test_dataloader.dataset)
        test_loss, correct = 0, 0

        for X, y in self.test_dataloader:
            pred = self.model(X.to(self.device))
            # test_loss += loss_fn(pred, y.to(self.device)).item()
            correct += (pred.argmax(1) == y.to(self.device)).type(torch.float).sum().item()
        correct /= size

        return correct

    def _test_SpeechCommand(self):
        # dataset_size = len(self.test_dataloader.dataset)
        correct = 0
        # for data, target in self.test_dataloader:
        #     data = data.to(self.device)
        #     target = target.to(self.device)
        #     # apply transform and model on whole batch directly on device
        #     data = self.transform(data)
        #     output = self.model(data)

        #     pred = get_likely_index(output)
        #     # pred = output.argmax(dim=-1)
        #     correct += number_of_correct(pred, target)

        # return 1.0 * correct / dataset_size

    def _test_AG_NEWS(self):
        self.model.eval()

        total_acc, total_count = 0, 0
        with torch.no_grad():
            for idx, (label, text, offsets) in enumerate(self.test_dataloader, 0):
                predicted_label = self.model(text, offsets)
                # loss = self.loss_fn(predicted_label, label)
                total_acc += (predicted_label.argmax(1) == label).sum().item()
                total_count += label.size(0)
            
        return total_acc/total_count