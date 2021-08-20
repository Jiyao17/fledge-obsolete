
import torch
from torch import nn, Tensor

from typing import List, Dict

from utils.audio import count_parameters
from utils.client import Client


class Server():
    def __init__(self,
            task: str,
            clients: List[Client],
            epoch_num: int,
            device: str="cpu",
            ):

        self.task = task
        self.clients = clients
        self.epoch_num = epoch_num
        self.device = device

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

    def test_model(self) -> float:
        pass