# Use PyTorch and network.py to train a FNN on the MNIST dataset

import torch
from torch import nn
from torch.utils.data import DataLoader
from torchvision import datasets
from torchvision.transforms import v2

from network import FullyConnectedNetwork

def train_model(filename: str = None):
    # Train model on MNIST digit dataset
    training_data = datasets.MNIST(
        root="dataset_data",
        train=True,
        download=True,
        transform=v2.Compose([v2.ToImage(), v2.ToDtype(torch.float32, scale=True)])
    )
    test_data = datasets.MNIST(
        root="dataset_data",
        train=False,
        download=True,
        transform=v2.Compose([v2.ToImage(), v2.ToDtype(torch.float32, scale=True)])
    )

    # Hyperparameter setup
    batch_size = 64
    learning_rate = 1e-3
    epochs = 20

    # Set up dataloaders, model, loss function and optimizer
    train_dataloader = DataLoader(training_data, batch_size=batch_size)
    test_dataloader = DataLoader(test_data, batch_size=batch_size)
    model = FullyConnectedNetwork([28*28, 512, 256, 10])
    loss_fn = nn.CrossEntropyLoss()
    optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate)

    # Train model
    for t in range(epochs):
        print(f"===== Epoch {t+1} =====")
        model.train_loop(train_dataloader, loss_fn, optimizer, 100)
        model.test_loop(test_dataloader, loss_fn)
        print("")
    print("===== Epochs Done =====")

    # Save model if filename supplied
    if filename is not None:
        model.export_params(filename)


def run_model(filename: str):
    # run pre-trained MNIST dataset model
    model = FullyConnectedNetwork([28*28, 512, 256, 10])
    model.load_from_params(filename)

    # TODO Add input loading and inference


# TODO Here, add a main to switch between train and run mode, export/import weights etc.

if __name__ == "__main__":
    train_model("model.pt")
    #run_model("model.pt")

