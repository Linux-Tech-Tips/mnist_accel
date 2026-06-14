from collections import OrderedDict

import torch
from torch import nn
from torch.utils.data import DataLoader
from torch.optim.optimizer import Optimizer
from torch.nn.modules.loss import _Loss


class FullyConnectedNetwork(nn.Module):
    """Fully Connected Network, ReLU activation on hidden layers"""

    def __init__(self, layers: list):
        """layers: list of numbers corresponding to layers and their neuron counts"""
        super().__init__()
        self.flatten = nn.Flatten()
        self.layers = nn.Sequential(
            OrderedDict(self._get_layer_list(layers))
        )

    def forward(self, x):
        x = self.flatten(x)
        logits = self.layers(x)
        return logits

    def train_loop(self, dataloader: DataLoader, loss_fn: _Loss, optimizer: Optimizer, print_interval: int = None):
        print("--- training loop ---")
        dataset_size = len(dataloader.dataset)
        batch_size = dataloader.batch_size
        # set model to train mode
        self.train()

        for batch, (X, y) in enumerate(dataloader):
            # Compute loss and prediction with forward pass computation graph
            prediction = self(X)
            loss = loss_fn(prediction, y)

            # Traverse computation graph of forward pass backwards to accumulate loss gradients
            loss.backward()
            # Use calculated gradients to perform a single optimizer step
            optimizer.step()
            # Reset gradients for next training step
            optimizer.zero_grad()

            if print_interval is not None:
                if batch % print_interval == 0:
                    loss, current = loss.item(), batch * batch_size + len(X)
                    print(f"loss: {loss:>7f} [{current:>5d}/{dataset_size:>5d}]")

        print("--- training loop done ---")

    def test_loop(self, dataloader: DataLoader, loss_fn: _Loss):
        print("--- test loop ---")
        dataset_size = len(dataloader.dataset)
        num_batches = len(dataloader)
        test_loss, correct = 0, 0
        # set model to eval mode
        self.eval()

        # evaluate without computing gradients/computation graph to save memory
        with torch.no_grad():
            for X, y in dataloader:
                prediction = self(X)
                test_loss += loss_fn(prediction, y).item()
                correct += (prediction.argmax(1) == y).type(torch.float).sum().item()

        test_loss /= num_batches
        correct /= dataset_size
        print(f"Test Result:\n -> accuracy: {100*correct}\n -> average loss: {test_loss}")
        print("--- test loop done ---")

    def export_params(self, path: str):
        """Saves model for inference using state_dict"""
        torch.save(self.state_dict(), path)

    def load_from_params(self, path: str):
        """Loads model for inference from given params/state_dict path"""
        self.load_state_dict(torch.load(path, weights_only=True))
        self.eval()

    def _get_layer_list(self, layers: list):
        # Initial layer
        result = [("fc0", nn.Linear(layers[0], layers[1]))]
        # Append ReLU for previous layer and next layer to leave final (output) layer without ReLU
        for i in range(1, len(layers) - 1):
            result.append((f"relu{i}", nn.ReLU()))
            result.append((f"dropout{i}", nn.Dropout(0.3)))
            result.append((f"fc{i}", nn.Linear(layers[i], layers[i+1])))
        return result

