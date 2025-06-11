import torch
from torch import nn

from torchvision import datasets
from torchvision.transforms import ToTensor

import matplotlib.pyplot as plt
from torch.utils.data import DataLoader

from pathlib import Path

### Loading our FashionMNIST dataset :-
train_data = datasets.FashionMNIST(
    root="data",
    train=True,
    download=False,
    transform=ToTensor(),
    target_transform=None
)
test_data = datasets.FashionMNIST(
    root="data",
    train=False,
    download=False,
    transform=ToTensor(),
    target_transform=None
)

### Pre-processing our dataset to convert in PyTorch DataLoader format and later into tensors :-
BATCH_SIZE = 32 
train_dataloader = DataLoader(dataset=train_data, batch_size=BATCH_SIZE, shuffle=True)
test_dataloader = DataLoader(dataset=test_data, batch_size=BATCH_SIZE, shuffle=False)
class_names = train_data.classes
print(class_names)

### Building our ResNet-18 Neural Network class :-
class ResNet_Model(nn.Module):
    def __init__(self, input_shape: int, hidden_units: int, output_shape: int):
        super().__init__()
        self.conv_block = nn.Sequential(
            nn.Conv2d(in_channels=input_shape,
                      out_channels=output_shape,
                      kernel_size=3,
                      stride=1,
                      padding=1),
            nn.BatchNorm2d(num_features=hidden_units),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2)
        )
        self.layer_1 = nn.Sequential(
            nn.Conv2d(in_channels=hidden_units,
                      out_channels=hidden_units,
                      kernel_size=3,
                      stride=1,
                      padding=1),
            nn.BatchNorm2d(num_features=hidden_units),
            nn.ReLU(),
            nn.Conv2d(in_channels=hidden_units,
                      out_channels=hidden_units,
                      kernel_size=3,
                      stride=1,
                      padding=1),
            nn.BatchNorm2d(num_features=hidden_units)
        )
        self.layer_2 = nn.Sequential(
            nn.Conv2d(in_channels=hidden_units,
                      out_channels=hidden_units,
                      kernel_size=3,
                      stride=1,
                      padding=1),
            nn.BatchNorm2d(num_features=hidden_units),
            nn.ReLU(),
            nn.Conv2d(in_channels=hidden_units,
                      out_channels=hidden_units,
                      kernel_size=3,
                      stride=1,
                      padding=1),
            nn.BatchNorm2d(num_features=hidden_units)
        )
        self.final_layer = nn.Sequential(
            nn.AdaptiveAvgPool2d((1,1)),
            nn.Flatten(),
            nn.Linear(in_features=hidden_units, out_features=output_shape)
        )

    def forward(self, x: torch.Tensor):
        x = self.final_layer(self.layer_2(self.layer_1(self.conv_block(x))))
        return x
    
torch.manual_seed(20)
model = ResNet_Model(input_shape=1, hidden_units=10, output_shape=len(class_names))

### Setting up loss_fn, optimizer and thier respective functions along with
### accuracy_fn(to know the current state of our data) to train our dataset :-
loss_fn = torch.nn.CrossEntropyLoss()
optimizer = torch.optim.SGD(params=model.parameters(), lr=0.1)

def accuracy_fn(y_true, y_pred):
    correct = torch.eq(y_true, y_pred).sum().item()
    acc = (correct / len(y_pred)) * 100
    return acc

def train_step(model: torch.nn.Module,
               data_loader: torch.utils.data.DataLoader,
               loss_fn: torch.nn.Module,
               optimizer: torch.optim.Optimizer,
               accuracy_fn):
    train_loss, train_acc = 0, 0
    model.train()

    for batch, (x,y) in enumerate(data_loader):
        y_pred = model(x)

        loss = loss_fn(y_pred, y)
        train_loss += loss
        acc = accuracy_fn(y_true=y, y_pred=y_pred.argmax(dim=1))
        train_acc += acc

        optimizer.zero_grad()

        loss.backward()

        optimizer.step()

    train_loss /= len(data_loader)
    train_acc /= len(data_loader)
    print(f"Train-loss: {train_loss:.5f} | Train-acc: {train_acc:.2f}%")

def print_train_time(start, end):
    total_time = end - start
    print(f"\nTrain time: {total_time:.3f} seconds")
    return total_time

from timeit import default_timer as timer
from tqdm.auto import tqdm

torch.manual_seed(20)
epochs = 4
strat_time = timer()

for epoch in tqdm(range(1,epochs+1)):
    print(f"\nEpoch: {epoch}\n--------")
    train_step(model, train_dataloader, loss_fn, optimizer, accuracy_fn)

end_time = timer()
total_time = print_train_time(strat_time, end_time)
print(model.state_dict())

### Saving our model :- 
MODEL_PATH = Path("models")
MODEL_PATH.mkdir(parents=True, exist_ok=True)
MODEL_NAME = "ResNet-Model.pt"
MODEL_SAVE_PATH = MODEL_PATH / MODEL_NAME
print("Saving model to : ", MODEL_SAVE_PATH)
torch.save(obj=model.state_dict(), f=MODEL_SAVE_PATH)