

from torch import nn

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
