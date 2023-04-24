# https://medium.com/mlearning-ai/vision-transformers-from-scratch-pytorch-a-step-by-step-guide-96c3313c2e0c

import numpy as np

import torch
import torch.nn as nn
from torch.optim import Adam
from torch.nn import CrossEntropyLoss
from torch.utils.data import DataLoader

from torchvision.transforms import ToTensor
from torchvision.datasets.mnist import MNIST

import timm

import einops

from models import unet, segmenter

np.random.seed(0)
torch.manual_seed(0)


def produce_label(x: torch.tensor) -> torch.tensor:
    y_seg = torch.argmax((x > 0.5).float(), dim=1).long()
    return y_seg


def main():
    # Loading data
    transform = ToTensor()

    train_set = MNIST(root='./../datasets', train=True, download=True, transform=transform)
    test_set = MNIST(root='./../datasets', train=False, download=True, transform=transform)

    train_loader = DataLoader(train_set, shuffle=True, batch_size=128)
    test_loader = DataLoader(test_set, shuffle=False, batch_size=128)

    # Defining model and training options
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Using device: ", device, f"({torch.cuda.get_device_name(device)})" if torch.cuda.is_available() else "")


    # model = unet.UNet(n_channels=1, n_classes=2)
    model = segmenter.Segmenter((1, 28, 28), decoder_type='mask_transformer_v2')

    N_EPOCHS = 2
    LR = 0.0001
    global_step = 0
    log_freq = 100
    loss_set = []

    # Training loop
    optimizer = Adam(model.parameters(), lr=LR)
    # criterion = CrossEntropyLoss()
    criterion = nn.CrossEntropyLoss()
    model.to(device)
    model.train()
    for epoch in range(N_EPOCHS):
        train_loss = 0.0
        for batch in train_loader:
            x, y = batch
            x, y = x.to(device), y.to(device)
            # x = x.repeat(1, 3, 1, 1)
            y_seg = produce_label(x)

            y_hat = model(x)
            loss = criterion(y_hat, y_seg)

            train_loss += loss.detach().cpu().item() / len(train_loader)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            global_step += 1
            loss_set.append(loss.item())
            if global_step % log_freq == 0:
                print(f'{np.mean(loss_set):.3f}')
                loss_set = []

        print(f"Epoch {epoch + 1}/{N_EPOCHS} loss: {train_loss:.2f}")

    # Test loop
    with torch.no_grad():
        correct, total = 0, 0
        test_loss = 0.0
        for batch in test_loader:
            x, y = batch
            x, y = x.to(device), y.to(device)
            x = x.repeat(1, 3, 1, 1)
            y_hat = model(x)
            loss = criterion(y_hat, y)
            test_loss += loss.detach().cpu().item() / len(test_loader)

            correct += torch.sum(torch.argmax(y_hat, dim=1) == y).detach().cpu().item()
            total += len(x)
        print(f"Test loss: {test_loss:.2f}")
        print(f"Test accuracy: {correct / total * 100:.2f}%")


if __name__ == '__main__':
    # print([m for m in timm.list_models() if 'vit' in m])
    main()
