
import torch
import json
import pickle

import socket


class ServerNet():
    def __init__(self, port, worker_num):
        self.sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)       
        self.sock.bind(('127.0.0.1', port))
        self.sock.listen(worker_num)

        self.worker_num = worker_num
        self.conn_list = []


    def connect_clients(self):
        worker_num = 0
        while worker_num < self.worker_num:
            conn, addr = self.sock.accept()
            self.conn_list.append(conn)

            worker_num += 1

    def recv(self, conn_num, length):
        msg = "".encode()
        while length > 0:
            msg  += self.conn_list[conn_num].recv(65536)
            length -= 65536

        return msg

    def __del__(self):
        for conn in self.conn_list:
            conn.close()


class Server():
    def __init__(self, config="config.json", model=None):
        try:
            with open(config, "r") as f:
                config = json.load(f)
        except:
            raise "Cannot open/parse config file."
        port = config["server"]["address"]["port"]
        client_num = config["clients"]["client_num"]
        self.net = ServerNet(port, client_num)

        if model == None:
            raise "Invalid model."
        self.model = model
        self.device = config["server"]["device"]
        self.model.to(self.device)
        self.model_state_dict = model.state_dict()
    
    def connect_clients(self):
        self.net.connect_clients()

    def send_model(self):
        """
        Send global model to clients.
        """
        state_bytes = pickle.dumps(self.model_state_dict)
        for net in self.net.conn_list:
            msg_len = len(state_bytes)
            # b = len(state_bytes).to_bytes(4, 'big')            
            # print(msg_len)
            # print(len(b))
            # print(int.from_bytes(b, "big"))
            net.send(len(state_bytes).to_bytes(4, 'big'))
            net.send(state_bytes)

    def update_model(self):
        """
        Server aggregates new models.
        """
        state_dict_list = []
        for index, net in enumerate(self.net.conn_list):
            dict_len = net.recv(4)
            state_bytes = self.net.recv(index, int.from_bytes(dict_len, 'big'))
            state_dict = pickle.loads(state_bytes)
            state_dict_list.append(state_dict)

        state_dict_avg = state_dict_list[0]
        for key in state_dict_avg.keys():
            for i in range(1, len(state_dict_list)):
                state_dict_avg[key] += state_dict_list[i][key]
            state_dict_avg[key] = torch.div(state_dict_avg[key], len(state_dict_list))
        
        self.model_state_dict = state_dict_avg
        self.model.load_state_dict(self.model_state_dict)
        self.model.to(self.device)

