
from math import pi
import socket
import json
import pickle
from numpy.core.records import array
from torch.optim.lr_scheduler import StepLR
from torch import nn
from torch.optim import Optimizer
from torch.nn.modules import loss
from torch.utils.data import DataLoader
from torchvision import datasets
from torchvision.transforms import ToTensor



class ClientNet():
    def __init__(self, server_addr: tuple):
        self.server_addr = server_addr
        
        self.sock = None
    
    def connect_to_server(self):
        self.sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        self.sock.connect(self.server_addr)

    def recv(self, length: int):
        msg = "".encode()
        while len(msg) < length:
            msg += self.sock.recv(length - len(msg))

        return msg
        
        # if length <= 50000:
        #     return self.sock.recv(length)
        # else:
        #     msg = "".encode()
        #     while length > 50000:
        #         # print("received %d bytes of %d bytes" % (received_len, recv_len))
        #         msg  += self.sock.recv(50000)
        #         length -= 50000
        #     msg += self.sock.recv(length)

        #     return msg

    def send(self, data: bytes):
        sent_len = 0
        data_len = len(data)

        while sent_len < data_len:
            sent_len += self.sock.send(data[sent_len:])

        return sent_len
        

    def recv_model(self, model: nn.Module=None):
        state_bytes_len_b = self.recv(4)
        state_bytes_len = int.from_bytes(state_bytes_len_b, 'big')
        # print("download model len: %d" % model_len)
        state_bytes = self.recv(state_bytes_len)
        print("client: recv model len: %d" % len(state_bytes))
        return state_bytes

    def send_model(self, model: nn.Module=None):
        # upload model
        state_dict = model.state_dict()
        state_bytes = pickle.dumps(state_dict)
        # print("uploading model with length: %d" % len(state_bytes))
        state_bytes_len_b = len(state_bytes).to_bytes(4, 'big')
        self.send(state_bytes_len_b)
        sent_len = self.send(state_bytes)
        print("client: send msg len: %d" % sent_len)

class Client():
    def __init__(self, 
            server_addr: tuple=("127.0.0.1", 5000),
            task: str="FashionMNIST",
            dataloader: DataLoader=None,
            model: nn.Module=None,
            loss_fn = None,
            optimizer: Optimizer=None,
            scheduler: StepLR=None,
            transform = None,
            epoch_num: int=5,
            device: str="cpu"
            ):
        if dataloader == None:
            raise "Invalid dataloader."
        if model == None:
            raise "Invalid model."
        if optimizer == None:
            raise "Invalid optimizer."

        self.net = ClientNet(server_addr)
        self.task = task
        self.dataloader = dataloader
        self.model = model.to(device)
        self.loss_fn = loss_fn
        self.optimizer=optimizer
        self.scheduler=scheduler
        self.transform=transform
        self.epoch_num = epoch_num
        self.device = device
        self.model_state_dict = self.model.state_dict()
        self.model_len = len(pickle.dumps(self.model_state_dict))
        raw_model_len = len(pickle.dumps(model.state_dict()))
        print("client: raw model len: %d" % raw_model_len)
        print("client: self.model len: %d" % self.model_len)

    def init(self):
        self.net.connect_to_server()

    def download_model(self):
        # print("Downloading model......")
        # get model
        # sd_len = net_list[j].recv(4)
        # sd_len = self.net.recv(4)
        # print("download model len: %d" % int.from_bytes(sd_len, 'big'))
        # print("download model len: %d" % self.model_len)
        state_bytes = self.net.recv(self.model_len)
        # print("Model downloaded.")
        state_dict = pickle.loads(state_bytes)
        # print("Loading model to GPU")
        self.model.load_state_dict(state_dict)
        self.model = self.model.to(self.device)

    def train_model(self):
        for i in range(self.epoch_num):
            # print(len(dataloader_list[j]))
            if self.task == "FashionMNIST":
                self.train_FashionMNIST()
            elif self.task == "SpeechCommand":
                self.train_SpeechCommand()
                self.scheduler.step()

    def upload_model(self):
        # upload model
        state_dict = self.model.state_dict()
        state_bytes = pickle.dumps(state_dict)
        # print("uploading model with length: %d" % len(state_bytes))
        # self.net.send(len(state_bytes).to_bytes(4, 'big'))
        self.net.send(state_bytes)

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
        