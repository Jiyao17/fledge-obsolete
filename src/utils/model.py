
from torch import nn
import torch.nn.functional as F
import torch


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


class SpeechCommand_Simplified(nn.Module):
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


class SpeechCommand_M5(nn.Module):
    def __init__(self, n_input=1, n_output=35, stride=16, n_channel=32):
        super().__init__()

        self.conv1 = nn.Conv1d(n_input, n_channel, kernel_size=80, stride=stride)
        self.bn1 = nn.BatchNorm1d(n_channel)
        self.pool1 = nn.MaxPool1d(4)
        self.conv2 = nn.Conv1d(n_channel, n_channel, kernel_size=3)
        self.bn2 = nn.BatchNorm1d(n_channel)
        self.pool2 = nn.MaxPool1d(4)
        self.conv3 = nn.Conv1d(n_channel, 2 * n_channel, kernel_size=3)
        self.bn3 = nn.BatchNorm1d(2 * n_channel)
        self.pool3 = nn.MaxPool1d(4)
        self.conv4 = nn.Conv1d(2 * n_channel, 2 * n_channel, kernel_size=3)
        self.bn4 = nn.BatchNorm1d(2 * n_channel)
        self.pool4 = nn.MaxPool1d(4)
        self.fc1 = nn.Linear(2 * n_channel, n_output)

    def forward(self, x):
        x = self.conv1(x)
        x = F.relu(self.bn1(x))
        x = self.pool1(x)
        x = self.conv2(x)
        x = F.relu(self.bn2(x))
        x = self.pool2(x)
        x = self.conv3(x)
        x = F.relu(self.bn3(x))
        x = self.pool3(x)
        x = self.conv4(x)
        x = F.relu(self.bn4(x))
        x = self.pool4(x)
        x = F.avg_pool1d(x, x.shape[-1])
        x = x.permute(0, 2, 1)
        x = self.fc1(x)
        return F.log_softmax(x, dim=2)


class AG_NEWS_TEXT(nn.Module):
    def __init__(self, vocab_size = 95811, embed_dim = 64, num_class = 4):
        super(AG_NEWS_TEXT, self).__init__()
        self.embedding = nn.EmbeddingBag(vocab_size, embed_dim, sparse=True)
        self.fc = nn.Linear(embed_dim, num_class)
        self.init_weights()

        from torchtext.data.utils import get_tokenizer
        from torchtext.vocab import build_vocab_from_iterator
        from torchtext.datasets import AG_NEWS

        train_iter = AG_NEWS(split='train')
        self.tokenizer = get_tokenizer('basic_english')
        self.vocab = build_vocab_from_iterator(self.yield_tokens(train_iter), specials=["<unk>"])
        self.vocab.set_default_index(self.vocab["<unk>"])
        self.text_pipeline = lambda x: self.vocab(self.tokenizer(x))
        self.label_pipeline = lambda x: int(x) - 1
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    def init_weights(self):
        initrange = 0.5
        self.embedding.weight.data.uniform_(-initrange, initrange)
        self.fc.weight.data.uniform_(-initrange, initrange)
        self.fc.bias.data.zero_()

    def forward(self, text, offsets):
        embedded = self.embedding(text, offsets)
        return self.fc(embedded)

    def yield_tokens(self, data_iter):
        for _, text in data_iter:
            yield self.tokenizer(text)
    
    def collate_batch(self, batch):
        label_list, text_list, offsets = [], [], [0]
        for (_label, _text) in batch:
            label_list.append(self.label_pipeline(_label))
            processed_text = torch.tensor(self.text_pipeline(_text), dtype=torch.int64)
            text_list.append(processed_text)
            offsets.append(processed_text.size(0))
        label_list = torch.tensor(label_list, dtype=torch.int64)
        offsets = torch.tensor(offsets[:-1]).cumsum(dim=0)
        text_list = torch.cat(text_list)
        return label_list.to(self.device), text_list.to(self.device), offsets.to(self.device)