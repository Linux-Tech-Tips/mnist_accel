# Use PyTorch and network.py to train a FNN on the MNIST dataset

import sys

import torch
from torch import nn
from torch.utils.data import DataLoader
from torchvision import datasets
from torchvision.transforms import v2

from PIL import Image

from network import FullyConnectedNetwork

BATCH_SIZE=64
LEARNING_RATE=1e-4
EPOCHS=20
MODEL=[28*28, 512, 256, 10]


def train_model(filename: str = None):
    # Train model on MNIST digit dataset
    training_data = datasets.MNIST(
        root="dataset_data",
        train=True,
        download=True,
        # Add data augmentation to transform to prevent overfitting to the otherwise clean MNIST
        transform=v2.Compose([
            v2.ToImage(), 
            v2.ToDtype(torch.float32, scale=True),
            v2.RandomRotation(degrees=10),
            v2.RandomAffine(degrees=0, translate=(0.1, 0.1))
        ])
    )
    test_data = datasets.MNIST(
        root="dataset_data",
        train=False,
        download=True,
        transform=v2.Compose([v2.ToImage(), v2.ToDtype(torch.float32, scale=True)])
    )

    # Hyperparameter setup
    batch_size = BATCH_SIZE
    learning_rate = LEARNING_RATE
    epochs = EPOCHS

    # Set up dataloaders, model, loss function and optimizer
    train_dataloader = DataLoader(training_data, batch_size=batch_size, num_workers=4, pin_memory=True)
    test_dataloader = DataLoader(test_data, batch_size=batch_size)
    model = FullyConnectedNetwork(MODEL)
    loss_fn = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

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


def run_model(filename: str, image_filename: str):
    # run pre-trained MNIST dataset model
    model = FullyConnectedNetwork(MODEL)
    model.load_from_params(filename)
    print(f"Loaded model from {filename}, running inference")

    image_transform = v2.Compose([v2.ToImage(), v2.ToDtype(torch.float32, scale=True)])
    image = Image.open(image_filename).convert("L")
    tensor = image_transform(image).unsqueeze(0)

    result = model(tensor)
    print(f"Inference run on {image_filename}, result:")
    print(torch.argmax(result).item())


if __name__ == "__main__":
    # Printing help if requested
    if "--help" in sys.argv or "-h" in sys.argv or "help" in sys.argv:
        print("Help:")
        print(f"> {sys.argv[0]} [train|infer] [model filename]")
        print(" - specifying filename requires 'train' or 'infer' to be specified in the first argument")
        print(" - not specifying filename or mode+filename will prompt for the missing options")
        exit(0)

    # Getting mode
    if len(sys.argv) < 2:
        mode = input("Please select mode: (train/infer) ")
    else:
        mode = sys.argv[1]

    # Getting model filename
    if len(sys.argv) < 3:
        filename = input("Please enter model filename (including extension, typically .pt): ")
    else:
        filename = sys.argv[2]

    # Running appropriate mode
    if mode.strip().lower() == "train":
        train_model(filename)
    elif mode.strip().lower() == "infer":
        # Getting image filename
        if len(sys.argv) < 4:
            image_file = input("Please enter name of image to run: ")
        else:
            image_file = sys.argv[3]
        run_model(filename, image_file)
    else:
        print(f"Unknown mode: {mode}")

