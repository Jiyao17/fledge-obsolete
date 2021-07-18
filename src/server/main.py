
import sys
import pickle
import torch
from torch import nn
from torch.utils.data import DataLoader
from torchvision import datasets
from torchvision.transforms import ToTensor

# federated tools
from utils.model import FashionMNIST_CNN
from utils.server import Server



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

    return correct

if __name__ == "__main__":

    client_num: int = int(sys.argv[1])
    EPOCH_NUM: int = int(sys.argv[2])
    task: str = sys.argv[3]

    result_dir = "result.txt"
    device = "cuda" if torch.cuda.is_available() else "cpu"

    if task == "FashionMNIST":
        test_dataset = datasets.FashionMNIST(
            root="~/fledge/data",
            train=False,
            download=True,
            transform=ToTensor()
            )
        model = FashionMNIST_CNN()
    elif task == "CIFAR":
        pass
    else:
        raise "Task not supported yet."
    
    test_dataloader = DataLoader(test_dataset, batch_size=64)
    model_len = len(pickle.dumps(model.state_dict()))
    print("raw model len: %d" % model_len)

    # config_file = "/home/jiyaoliu17/fledge/src/server/config.json"
    server = Server(client_num=client_num, model=model, device="cuda")
    print("Server initialized")
    server.init_client_net()
    print("Clients connected")

    f = open(result_dir, "a+")
    f.write("\n")
    for i in range(EPOCH_NUM):
        print("Epoch %d......" % i)

        # print("Sending model to clients......")
        server.distribute_model()
        # print("Receiving new models and updating......")
        
        server.aggregate_model()
        # print("Testing new model......")
        correct = test_loop(test_dataloader, server.model, torch.nn.CrossEntropyLoss())

        if i % 10 == 9:
            f.write(f"{(100*correct):>0.1f}% ")

    f.close()


