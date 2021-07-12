
import socket
import json
import pickle

from torch import nn
from torch.optim import Optimizer
from torch.nn.modules import loss
from torch.utils.data import DataLoader
from torchvision import datasets
from torchvision.transforms import ToTensor


class ClientNet():
    def __init__(self, config_file="config.json"):
        try:
            with open(config_file, "r") as f:
                config = json.load(f)
        except:
            raise "Cannot open/parse config file."
        
        try:
            server_ip = config["server"]["address"][0]
            server_port = config["server"]["address"][1]
        except:
            raise "Invalid config file."

        self.server_addr = ((server_ip, server_port))
    
    def connect_to_server(self):
        self.sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        self.sock.connect(self.server_addr)

    def recv(self, length):
        if length <= 10000:
            return self.sock.recv(length)
        else:
            msg = "".encode()
            while length > 10000:
                # print("received %d bytes of %d bytes" % (received_len, recv_len))
                msg  += self.sock.recv(10000)
                length -= 10000
            msg += self.sock.recv(length)

            return msg

    def send(self, data):
        return self.sock.send(data)

class Client():
    def __init__(self, 
            config_file="config.json",
            dataloader:DataLoader=None,
            model:nn.Module=None,
            loss_fn=nn.CrossEntropyLoss(),
            optimizer:Optimizer=None
            ):
        try:
            with open(config_file, "r") as f:
                config = json.load(f)
        except:
            raise "Cannot open/parse config file."
        if dataloader == None:
            raise "Invalid dataloader."
        if model == None:
            raise "Invalid model."

        self.device:str = config["self"]["device"]
        self.net = ClientNet(config_file)
        self.dataloader = dataloader
        self.model = model
        self.loss_fn = loss_fn
        self.optimizer=optimizer

    def init(self):
        self.net.connect_to_server()

    def download_model(self):
        # print("Downloading model......")
        # get model
        # sd_len = net_list[j].recv(4)
        sd_len = self.net.recv(4)
        print(int.from_bytes(sd_len, 'big'))
        state_bytes = self.net.recv(int.from_bytes(sd_len, 'big'))
        # print("Model downloaded.")
        state_dict = pickle.loads(state_bytes)
        # print("Loading model to GPU")
        self.model.load_state_dict(state_dict)
        self.model.to(self.device)

    def train_model(self):
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
        self.net.send(len(state_bytes).to_bytes(4, 'big'))
        self.net.send(state_bytes)

        