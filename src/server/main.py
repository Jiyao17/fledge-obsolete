
import sys
import pickle
import torch
from torch import nn
from torch.utils.data import DataLoader, dataloader
from torchvision import datasets
from torchvision.transforms import ToTensor
import torchaudio

# federated tools
# sys.path.append("~/projects/fledge/utils")
from utils.model import FashionMNIST_CNN, SpeechCommand_M5
from utils.server import Server
from utils.audio import SubsetSC, collate_fn, number_of_correct, get_likely_index, set_LABELS, count_parameters

def test_MNIST(dataloader, model, loss_fn, device):
    size = len(dataloader.dataset)
    test_loss, correct = 0, 0

    with torch.no_grad():
        for X, y in dataloader:
            pred = model(X.to(device))
            test_loss += loss_fn(pred, y.to(device)).item()
            correct += (pred.argmax(1) == y.to(device)).type(torch.float).sum().item()
            
    correct /= size
    print(f"Accuracy: {(100*correct):>0.1f}%\n")

    return correct

def test_audio(dataloader, model, transform, device):
    model.eval()
    correct = 0
    for data, target in dataloader:
        data = data.to(device)
        target = target.to(device)
        # apply transform and model on whole batch directly on device
        data = transform(data)
        output = model(data)

        pred = get_likely_index(output)
        correct += number_of_correct(pred, target)

    print(f"Accuracy: {correct}/{len(dataloader.dataset)} ({100. * correct / len(dataloader.dataset):.0f}%)\n")
    return correct / len(dataloader.dataset)

if __name__ == "__main__":
    client_num: int = int(sys.argv[1])
    EPOCH_NUM: int = int(sys.argv[2])
    task: str = sys.argv[3]

    result_dir = "result.txt"
    device = "cuda" if torch.cuda.is_available() else "cpu"

    if task == "FashionMNIST":
        test_dataset = datasets.FashionMNIST(
            root="~projects/fledge/data/",
            train=False,
            download=True,
            transform=ToTensor()
            )
        test_dataloader = DataLoader(test_dataset, batch_size=64, drop_last=True)
        model = FashionMNIST_CNN()
    elif task == "SpeechCommand":
        test_dataset = SubsetSC("testing")
        if device == "cuda":
            num_workers = 1
            pin_memory = True
        else:
            num_workers = 0
            pin_memory = False
            
        test_dataloader = torch.utils.data.DataLoader(
            test_dataset,
            batch_size=64,
            shuffle=False,
            drop_last=True,
            collate_fn=collate_fn,
            num_workers=num_workers,
            pin_memory=pin_memory,
            )

        waveform, sample_rate, label, speaker_id, utterance_number = test_dataset[0]
        labels = sorted(list(set(datapoint[2] for datapoint in test_dataset)))
        set_LABELS(labels)
        new_sample_rate = 8000
        transform = torchaudio.transforms.Resample(orig_freq=sample_rate, new_freq=new_sample_rate)
        transformed = transform(waveform)
        model = SpeechCommand_M5(
            n_input=transformed.shape[0],
            n_output=len(labels)
            )
    else:
        raise "Task not supported yet."
    
    # model_len = len(pickle.dumps(model.state_dict()))
    # print("raw model len: %d" % model_len)

    # config_file = "/home/jiyaoliu17/fledge/src/server/config.json"
    server = Server(client_num=client_num, model=model, device=device)
    print("Server initialized")
    server.init_client_net()
    print("Clients connected")

    f = open(result_dir, "a+")
    # f.write("\n")
    for i in range(EPOCH_NUM):
        print("Epoch %d......" % i)

        # print("Sending model to clients......")
        server.distribute_model()
        # print("Receiving new models and updating......")
        
        server.aggregate_model()
        # print("Testing new model......")

        if task == "FashionMNIST":
            rate = test_MNIST(test_dataloader, server.model, torch.nn.CrossEntropyLoss(), device)
        elif task == "SpeechCommand":
            rate = test_audio(test_dataloader, server.model, transform, device)

        if i % 10 == 9:
            f.write(f"{(100*rate):>0.1f}% ")

    # f.write("\n")
    f.close()



