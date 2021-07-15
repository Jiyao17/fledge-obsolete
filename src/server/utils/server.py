
import torch
from torch import nn, Tensor
import json
import pickle

import socket

from typing import List, Dict


class ServerNet():
    def __init__(self, config_file="config.json"):
        try:
            with open(config_file, "r") as f:
                config = json.load(f)
        except:
            raise "Cannot open/parse config file."
        
        try:
            self.ip = config["self"]["address"][0]
            self.port = config["self"]["address"][1]
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

        self.send_flag = 0

    def init_net(self):
        """
        all servers in the network should do this before any network action
        """
        self.sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)       
        self.sock.bind((self.ip, self.port))
        self.sock.listen(self.server_num + self.client_num)

    def connect_servers(self):
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
        if length <= 10000:
            return conn.recv(length)
        else:
            msg = "".encode()
            while length > 10000:
                # print("received %d bytes of %d bytes" % (received_len, recv_len))
                msg  += conn.recv(10000)
                length -= 10000
            msg += conn.recv(length)

            return msg

    @staticmethod
    def send(conn: socket.socket, data):
        sent_len = 0
        data_len = len(data)

        while sent_len < data_len:
            sent_len += conn.send(data[sent_len:])

        


class Server():
    def __init__(self, config_file="config.json", model=None):
        try:
            with open(config_file, "r") as f:
                config = json.load(f)
        except:
            raise "Cannot open/parse config file."

        # init learning model
        if model == None:
            raise "Invalid model."
        self.model: nn.Module = model
        self.device = config["self"]["device"]
        self.model.to(self.device)
        self.model_state_dict = self.model.state_dict()
        self.model_len = len(pickle.dumps(self.model_state_dict))
        print("model len: %d" % self.model_len)


        # init net functions
        self.net = ServerNet(config_file)
    
    def init_server_net(self):
        """
        not useful in current work
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
        for conn in self.net.client_conn_list:
            # msg_len = len(state_bytes)
            # b = len(state_bytes).to_bytes(4, 'big')            
            # print(msg_len)
            # print(len(b))
            # print(int.from_bytes(b, "big"))
            conn.send(len(state_bytes).to_bytes(4, 'big'))
            conn.send(state_bytes)

    def aggregate_model(self):
        """
        Server aggregates new models.
        """
        # collect models from clients
        state_dict_list: List[Dict[str, Tensor]] = []
        for conn in self.net.client_conn_list:
            dict_len = ServerNet.recv(conn, 4)
            len_int = int.from_bytes(dict_len, 'big')
            print("Model length: %d" % len_int)
            state_bytes = ServerNet.recv(conn, int.from_bytes(dict_len, 'big'))
            state_dict: Dict[str, Tensor] = pickle.loads(state_bytes)
            state_dict_list.append(state_dict) # optimizable

        # calculate average model
        state_dict_avg = state_dict_list[0]
        for key in state_dict_avg.keys():
            for i in range(1, len(state_dict_list)):
                state_dict_avg[key] += state_dict_list[i][key]
            state_dict_avg[key] = torch.div(state_dict_avg[key], len(state_dict_list))
        
        # load average model
        self.model_state_dict = state_dict_avg
        self.model.load_state_dict(self.model_state_dict)
        self.model.to(self.device)

    def server_sync(self):
        pass
