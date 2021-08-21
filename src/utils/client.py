
from torch.optim.lr_scheduler import StepLR
from torch import nn
from torch.optim import Optimizer
from torch.nn.modules import loss
from torch.utils.data import DataLoader
from torchvision import datasets
from torchvision.transforms import ToTensor

from utils.model import FashionMNIST_CNN, SpeechCommand_M5


class Client():
    def __init__(self, 
            task: str,
            dataset: Dataset,
            epoch_num: int=5,
            device: str="cpu"
            ):
        self.task = task
        self.dataset = dataset
        self.epoch_num = epoch_num
        self.device = device

        self.model = self.init_model()

    def init_model(self) -> nn.Module:
        if self.task == "FashionMNIST":
            model = FashionMNIST_CNN()
        elif self.task == "SpeechCommand":
            model = SpeechCommand_M5()
        
        return model

    def train_model(self):
        for i in range(self.epoch_num):
            # print(len(dataloader_list[j]))
            if self.task == "FashionMNIST":
                self.train_FashionMNIST()
            elif self.task == "SpeechCommand":
                self.train_SpeechCommand()
                self.scheduler.step()

    def train_FashionMNIST(self):
        for batch, (X, y) in enumerate(self.dataloader):
            # Compute prediction and loss
            pred = self.model(X.cuda())
            loss = self.loss_fn(pred, y.cuda())
            
            # Backpropagation
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

    def train_SpeechCommand(self):
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
        
    def test_model(self) -> float:
        pass