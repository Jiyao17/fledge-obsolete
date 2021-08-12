

from torch import nn
import torch.nn.functional as F

class FashionMNIST_CNN(nn.Module):
    def __init__(self):
        super(FashionMNIST_CNN, self).__init__()

        self.net = nn.Sequential(
            nn.Conv2d(1, 12, kernel_size=5),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),
            nn.Conv2d(12, 24, 5),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),

            nn.Flatten(),

            nn.Linear(24*4*4, 50),
            nn.ReLU(),
            nn.Linear(50, 10),
            nn.ReLU(),
        )

    def forward(self, x):
        x = self.net(x)
        return x


class SpeechCommand_M5(nn.Module):
    def __init__(self, n_input=1, n_output=35, stride=16, n_channel=32):
        super().__init__()

        self.net = nn.Sequential(
            # 1*8000
            nn.Conv1d(n_input, n_channel, kernel_size=80, stride=stride),
            # 32*496
            nn.BatchNorm1d(n_channel),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=4, stride=1),
            # 32*493

            nn.Conv1d(n_channel, n_channel//2, kernel_size=3),
            # 16*491
            nn.BatchNorm1d(n_channel//2),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=4, stride=1),
            # 16*488

            nn.Conv1d(n_channel//2, n_channel//2, kernel_size=3),
            # 16*486
            nn.BatchNorm1d(n_channel//2),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=4, stride=1),
            # 16*483

            nn.Flatten(),

            nn.Linear(16*483, 512),
            nn.Linear(512, n_output),
            nn.LogSoftmax(dim=1)
        )



    def forward(self, x):
        
        x = self.net(x)
        return x