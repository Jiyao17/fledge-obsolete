
from torch.optim.lr_scheduler import StepLR
from torch import nn, optim
from torch.optim import Optimizer
from torch.nn.modules import loss
from torch.utils.data import Dataset, DataLoader
from torchaudio.transforms import Resample
from torchvision import datasets
from torchvision.transforms import ToTensor
from torch.nn.modules.loss import CrossEntropyLoss
import torchaudio
import torch.nn.functional as F


from utils.model import FashionMNIST_CNN, SpeechCommand_M5
from utils.audio import collate_fn, set_LABELS

class Client():
    def __init__(self, 
            task: str,
            test_dataset: Dataset,
            epoch_num: int=5,
            batch_size: int=256,
            lr: int=0.01,
            device: str="cpu"
            ):
        self.task = task
        self.test_dataset = test_dataset
        self.epoch_num = epoch_num
        self.batch_size=batch_size
        self.lr=lr
        self.device = device
        # set by self.init_task()
        self.train_dataloader: DataLoader = None
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
        else:
            raise "Unsupported task."

    def train_model(self):
        for i in range(self.epoch_num):
            # print(len(dataloader_list[j]))
            if self.task == "FashionMNIST":
                self._train_FashionMNIST()
            elif self.task == "SpeechCommand":
                self._train_SpeechCommand()
                self.scheduler.step()
     
    def test_model(self) -> float:
        pass

    def _init_FashionMNIST(self):
        self.train_dataloader = DataLoader(
            self.test_dataset,
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
            self.test_dataset,
            batch_size=self.batch_size,
            shuffle=True,
            collate_fn=collate_fn,
            num_workers=num_workers,
            pin_memory=pin_memory,
            drop_last=True
            )
        waveform, sample_rate, label, speaker_id, utterance_number = self.test_dataset[0]
        labels = sorted(list(set(datapoint[2] for datapoint in self.test_dataset)))
        set_LABELS(labels)
        new_sample_rate = 8000
        self.transform = torchaudio.transforms.Resample(orig_freq=sample_rate, new_freq=new_sample_rate)
        transformed: Resample = self.transform(waveform)
        self.loss_fn = F.nll_loss
        self.model = SpeechCommand_M5(
            n_input=transformed.shape[0],
            n_output=len(labels)
            )
        self.optimizer = optim.Adam(self.model.parameters(), lr=self.lr, weight_decay=0.0001)
        self.scheduler = optim.lr_scheduler.StepLR(self.optimizer, step_size=20, gamma=0.1)  # reduce the learning after 20 epochs by a factor of 10
    


    def _train_FashionMNIST(self):
        for batch, (X, y) in enumerate(self.dataloader):
            # Compute prediction and loss
            pred = self.model(X.cuda())
            loss = self.loss_fn(pred, y.cuda())
            
            # Backpropagation
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

    def _train_SpeechCommand(self):
        self.model.train()
        for batch_idx, (data, target) in enumerate(self.dataloader):
            data = data.to(self.device)
            target = target.to(self.device)

            # apply transform and model on whole batch directly on device
            # self.transform = self.transform.to(self.device)
            data = self.transform(data)
            output = self.model(data)

            # negative log-likelihood for a tensor of size (batch x 1 x n_output)
            loss = self.loss_fn(output.squeeze(), target)

            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

