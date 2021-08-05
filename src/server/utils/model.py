

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
            nn.MaxPool1d(kernel_size=17, stride=1),
            # 32*480
            nn.Conv1d(n_channel, 2*n_channel, kernel_size=3),
            # 64*478
            nn.BatchNorm1d(n_channel),
            nn.ReLU(),

            nn.Conv1d(n_channel, 2*n_channel, kernel_size=3),
            # 64*478
            nn.BatchNorm1d(n_channel),
            nn.ReLU(),

            nn.AvgPool1d(kernel_size=5, stride=1),
            # 16 26

            nn.Flatten(),

            nn.Linear(16*26, n_output),
            nn.ReLU(),
            nn.Softmax()
        )

    def forward(self, x):
        x = self.net(x)
        return x