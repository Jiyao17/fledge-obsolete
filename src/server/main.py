
import torch
from torch import nn
from torch.utils.data import DataLoader
from torchvision import datasets
from torchvision.transforms import ToTensor


# federated tools
from utils.model import FashionMNIST_CNN
from utils.server import ServerNet, Server

EPOCH_NUM = 40


def test_loop(dataloader, model, loss_fn):
    size = len(dataloader.dataset)
    test_loss, correct = 0, 0

    with torch.no_grad():
        for X, y in dataloader:
            pred = model(X.cuda())
            test_loss += loss_fn(pred, y.cuda()).item()
            correct += (pred.argmax(1) == y.cuda()).type(torch.float).sum().item()
            
    correct /= size
    print(f"Accuracy: {(100*correct):>0.1f}%\n")

if __name__ == "__main__":
    device = "cuda" if torch.cuda.is_available() else "cpu"
    test_dataset = datasets.FashionMNIST(
        root="../data",
        train=False,
        download=True,
        transform=ToTensor()
        )
    test_dataloader = DataLoader(test_dataset, batch_size=64)

    model = FashionMNIST_CNN()
    config_file = "/home/jiyaoliu17/fledge/src/server/config.json"
    server = Server(config_file, model)
    print("Server initialized")
    server.init_client_net()
    print("Clients connected")

    for i in range(EPOCH_NUM):
        print("Epoch %d......" % i)

        print("Sending model to clients......")
        server.distribute_model()
        print("Receiving new models and updating......")
        
        server.aggregate_model()
        print("Testing new model......")
        test_loop(test_dataloader, server.model, torch.nn.CrossEntropyLoss())



    # net = ServerNet(5000, 3)
    # net.connect_clients()
    # for index, conn in enumerate(net.conn_list):
    #     msg = conn.recv(1024)
    #     print("Message from client %d: %s" % (index, msg.decode()))
    #     msg = "Message %d received." % (index)
    #     conn.send(msg.encode())


