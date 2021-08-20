
import torch
from torch import nn, Tensor
import json
import pickle

import socket

from typing import List, Dict

from utils.audio import count_parameters

class ServerNet():
    def __init__(self, single=True, client_num=3,
            addr=("127.0.0.1", 5000),
            config_file: str= None
            ):

        self.single = single # single server or multiple servers
        self.client_num = client_num
        self.addr = addr
        if self.single == False:
            try:
                with open(config_file, "r") as f:
                    config = json.load(f)
            except:
                raise "Cannot open/parse config file."
            
            try:
                self.addr = tuple(config["self"]["address"])
                self.index = config["self"]["index"]

                self.server_num = config["servers"]["number"]
                self.server_addresses = config["servers"]["addresses"]
                self.server_topology = config["servers"]["topology"]

                self.client_num = config["clients"]["number"]
            except:
                raise "Invalid config file."

            # init by: init_net, connect_servers, connect_clients
            self.server_conn_list = [None for i in range(self.server_num)]
        
        self.sock = None
        self.client_conn_list: List[socket.socket] = []

    def init_net(self):
        """
        all servers in the network should do this before any network action
        """
        self.sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        self.sock.bind(self.addr)
        listen_num = 0
        if self.single: 
            listen_num = self.client_num
        else:
            listen_num = self.client_num + self.server_num
        self.sock.listen(listen_num)

    def connect_servers(self):
        r"""
        Not used or tested yet
        """
        # connect to other servers
        # servers should call this function in the order of their indexes
        i = self.index + 1
        while i < self.server_num:
            if self.server_topology[self.index][i] == 1:
                sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
                sock.connect(tuple(self.server_addresses[i]))
                self.server_conn_list[i] = sock

        # accept connections from other servers
        i = 0
        while i < self.index:
            if self.server_topology[self.index][i] == 1:
                conn, addr = self.sock.accept()
                self.server_conn_list[i] = conn

    def connect_clients(self):
        while len(self.client_conn_list) < self.client_num:
            conn, addr = self.sock.accept()
            self.client_conn_list.append(conn)

    @staticmethod
    def recv(conn: socket.socket, length):
        msg = "".encode()
        while len(msg) < length:
            msg += conn.recv(length - len(msg))

        print("server: recv msg len: %d" % len(msg))
        return msg

        # if length <= 50000:
        #     return conn.recv(length)
        # else:
        #     msg = "".encode()
        #     while length > 50000:
        #         # print("received %d bytes of %d bytes" % (received_len, recv_len))
        #         msg  += conn.recv(50000)
        #         length -= 50000
        #     msg += conn.recv(length)

        #     return msg

    @staticmethod
    def send(conn: socket.socket, data):
        sent_len = 0
        data_len = len(data)

        while sent_len < data_len:
            sent_len += conn.send(data[sent_len:])
        
        print("server: send msg len: %d" % len(data))


class Server():
    def __init__(self,
            single: bool=True,  client_num: int=3,
            model: nn.Module=None, device: str="cpu",
            config_file="config.json"
            ):

        self.single = single
        if model == None:
            raise "Invalid model."
        self.model = model.to(device)
        # self.model_len = len(pickle.dumps(self.model.state_dict()))
        # print("self.model len in mem: %d" % self.model_len)
        self.device = device
        # self.model_len = len(pickle.dumps(self.model.state_dict()))
        # print("self.model len in cuda: %d" % self.model_len)
        self.model_state_dict = self.model.state_dict()
        self.model_len = len(pickle.dumps(self.model_state_dict))
        raw_model_len = len(pickle.dumps(model.state_dict()))
        print("client: raw model len: %d" % raw_model_len)

        print("server: self.model len: %d" % self.model_len)

        # init net functions
        if self.single:
            net = ServerNet(client_num=client_num)
        else:
            net = ServerNet(single=False, config_file=config_file)
        self.net = net
    
    def init_server_net(self):
        """
        not used nor tested yet
        """
        self.net.init_net()
        self.net.connect_servers()

    def init_client_net(self):
        self.net.init_net()
        self.net.connect_clients()

    def distribute_model(self):
        """
        Send global model to clients.
        """
        state_bytes = pickle.dumps(self.model_state_dict)
        # print("length of model to distribute: %d" % len(state_bytes))
        for conn in self.net.client_conn_list:
            # msg_len = len(state_bytes)
            # b = len(state_bytes).to_bytes(4, 'big')            
            # print(msg_len)
            # print(len(b))
            # print(int.from_bytes(b, "big"))
            # conn.send(len(state_bytes).to_bytes(4, 'big'))
            ServerNet.send(conn, state_bytes)
            # conn.send(state_bytes)

    def aggregate_model(self):
        """
        Server aggregates new models.
        """
        # collect models from clients
        state_dict_list: List[Dict[str, Tensor]] = []
        for conn in self.net.client_conn_list:
            # dict_len = ServerNet.recv(conn, 4)
            # len_int = int.from_bytes(dict_len, 'big')
            # print("Model length: %d" % len_int)
            state_bytes = ServerNet.recv(conn, self.model_len)
            state_dict: Dict[str, Tensor] = pickle.loads(state_bytes)
            state_dict_list.append(state_dict) # optimizable

        # calculate average model
        state_dict_avg = state_dict_list[0]
        for key in state_dict_avg.keys():
            for i in range(1, len(state_dict_list)):
                state_dict_avg[key] += state_dict_list[i][key]
            state_dict_avg[key] = torch.div(state_dict_avg[key], len(state_dict_list))
        
        # load average model
        self.model.load_state_dict(state_dict_avg)
        self.model = self.model.to(self.device)
        self.model_state_dict = self.model.state_dict()
        # print("model len after aggregation: %d" % len(pickle.dumps(self.model.state_dict())))


class SServer(Server):
    def __init__(self, model):
        super().__init__(model=model)


class MServer(Server):
    def __init__(self, model, config_file):
        super().__init__(model=model)
        
        try:
            with open(config_file, "r") as f:
                config = json.load(f)
        except:
            raise "Cannot open/parse config file."
