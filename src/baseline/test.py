
import torch
from torch import optim
from torch.nn.modules.loss import CrossEntropyLoss
from torch.utils.data import DataLoader, random_split
from torchvision import datasets
from torchvision.transforms import ToTensor

# federated tools
from utils.model import FashionMNIST_CNN
from utils.audio import SubsetSC

EPOCH_NUM = 50
result_dir = "audio_result.txt"
lr = 0.01


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

    dataset_ratio = 0.5
    # ./test.sh 5 100 5 0.01 5000 FashionMNIST

    device = "cuda" if torch.cuda.is_available() else "cpu"
    test_dataset = SubsetSC("testing")
    print("test set length: %d" % len(test_dataset))
    test_dataloader = DataLoader(test_dataset, batch_size=64)
    train_dataset = SubsetSC("training")
    print("train set length: %d" % len(train_dataset))
    
    data_num = len(train_dataset)*dataset_ratio
    subset = random_split(train_dataset, [data_num, len(train_dataset)-data_num])[0]
    print("training dataset size: %d" % len(subset))
    train_dataloader = DataLoader(subset, batch_size=32)

    model = FashionMNIST_CNN().to(device)
    loss_fn = CrossEntropyLoss()
    optimizer = optim.SGD(model.parameters(), lr=lr)

    f = open(result_dir, "a+")
    f.write("\n")
    for epoch in range(EPOCH_NUM):
        for batch, (X, y) in enumerate(train_dataloader):
            # Compute prediction and loss
            pred = model(X.cuda())
            loss = loss_fn(pred, y.cuda())
            
            # Backpropagation
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        correct = test_loop(train_dataloader, model, loss_fn)   
        if epoch % 10 == 9:
            f.write(f"{(100*correct):>0.1f}% ")

    f.close()