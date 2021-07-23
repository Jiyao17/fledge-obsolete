
import socket
import json
import pickle
from numpy.core.records import array

from torch import nn
from torch.optim import Optimizer
from torch.nn.modules import loss
from torch.utils.data import DataLoader
from torchvision import datasets
from torchvision.transforms import ToTensor


class ClientNet():
    def __init__(self, server_addr: tuple):
        self.server_addr = server_addr
        
        self.send_flag = 0
    
    def connect_to_server(self):
        self.sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        self.sock.connect(self.server_addr)

    def recv(self, length):
        if length <= 50000:
            return self.sock.recv(length)
        else:
            msg = "".encode()
            while length > 50000:
                # print("received %d bytes of %d bytes" % (received_len, recv_len))
                msg  += self.sock.recv(50000)
                length -= 50000
            msg += self.sock.recv(length)

            return msg

    def send(self, data):
        sent_len = 0
        data_len = len(data)

        while sent_len < data_len:
            sent_len += self.sock.send(data[sent_len:])

class Client():
    def __init__(self, 
            server_addr: tuple=("127.0.0.1", 5000),
            dataloader: DataLoader=None,
            model: nn.Module=None,
            loss_fn = nn.CrossEntropyLoss(),
            optimizer: Optimizer=None,
            epoch_num: int=1,
            device: str="cpu"
            ):
        if dataloader == None:
            raise "Invalid dataloader."
        if model == None:
            raise "Invalid model."
        if optimizer == None:
            raise "Invalid optimizer."

        self.net = ClientNet(server_addr)
        self.dataloader = dataloader
        self.model = model
        self.loss_fn = loss_fn
        self.optimizer=optimizer
        self.epoch_num = epoch_num
        self.device = device
        self.model.to(self.device)
        self.model_state_dict = self.model.state_dict()
        self.model_len = len(pickle.dumps(self.model_state_dict))
        # print("self.model len: %d" % self.model_len)

    def init(self):
        self.net.connect_to_server()

    def download_model(self):
        # print("Downloading model......")
        # get model
        # sd_len = net_list[j].recv(4)
        # sd_len = self.net.recv(4)
        # print("download model len: %d" % int.from_bytes(sd_len, 'big'))
        state_bytes = self.net.recv(self.model_len)
        # print("Model downloaded.")
        state_dict = pickle.loads(state_bytes)
        # print("Loading model to GPU")
        self.model.load_state_dict(state_dict)
        self.model.to(self.device)

    def train_model(self):
        for i in range(self.epoch_num):
            # print(len(dataloader_list[j]))
            for batch, (X, y) in enumerate(self.dataloader):
                # Compute prediction and loss
                pred = self.model(X.cuda())
                loss = self.loss_fn(pred, y.cuda())
                
                # Backpropagation
                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()

    def upload_model(self):
        # upload model
        state_dict = self.model.state_dict()
        state_bytes = pickle.dumps(state_dict)
        # print("uploading model with length: %d" % len(state_bytes))
        # self.net.send(len(state_bytes).to_bytes(4, 'big'))
        self.net.send(state_bytes)

        