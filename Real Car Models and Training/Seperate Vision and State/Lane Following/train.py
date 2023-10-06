import torch
import torchvision
from torch import nn
from torch.utils.data import DataLoader, Dataset, SubsetRandomSampler
from torchvision import datasets
from torchvision.transforms import ToTensor, Lambda, Compose
import matplotlib.pyplot as plt
from pathlib import Path
import struct
import os
import torch.optim as optim
import torch.nn.functional as F
import numpy as np
import time
import cv2 as cv
import config
from dataset import CustomImageDataset
from model import Model

device = "cuda" if torch.cuda.is_available() else "cpu"

dataset = CustomImageDataset()

train_size = int(0.8 * len(dataset))
test_size = len(dataset) - train_size
train_dataset, test_dataset = torch.utils.data.random_split(dataset, [train_size, test_size])
train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=config.batch_size, shuffle=True)
test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=config.batch_size, shuffle=True)

model = Model().to(device)
test = torch.randn(16, config.input_channels, 256, 256).to(device)
print(model(test).shape)


loss_function = nn.MSELoss(reduction="sum")
optimizer = torch.optim.Adam(model.parameters(), lr=config.learning_rate)

batches_in_train = len(train_loader)
batches_in_test = len(test_loader)

for epoch in range(config.number_of_epochs):
    total_losses = 0.0
    train_images_counted = 0
    train_total_absolute_losses = 0
    train_total_in_range = []
    for limit in config.loss_cutoff_for_accuracy:
        train_total_in_range.append(0)
    for i, (images, ground_truths) in enumerate(train_loader):
        images = images.to(device)
        ground_truths = ground_truths.to(device)
        outputs = model(images)
        loss = loss_function(outputs, ground_truths.unsqueeze(1))
        total_losses += loss
        train_images_counted += ground_truths.shape[0]
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        train_loss_values_over_batch = abs(outputs.squeeze(1) - ground_truths)
        for x in train_loss_values_over_batch:
            train_total_absolute_losses = train_total_absolute_losses + x
        for index, limit in enumerate(config.loss_cutoff_for_accuracy):
            train_total_in_range[index] += len(train_loss_values_over_batch[train_loss_values_over_batch < limit])

    average_absolute_loss = train_total_absolute_losses / train_images_counted
    average_loss = total_losses / train_images_counted
    print(f"Epoch: {epoch} Loss: {average_loss}")
    print(f"Train Absolute Loss {average_absolute_loss}")
    for index, limit in enumerate(config.loss_cutoff_for_accuracy):
        print("Train Accuracy within limit: " + str(limit) + ": " + str(train_total_in_range[index] / train_images_counted))
    with torch.no_grad():
        test_total_counted = 0
        test_total_in_range = []
        test_total_absolute_losses = 0.0

        for limit in config.loss_cutoff_for_accuracy:
            test_total_in_range.append(0)
        for i, (test_images, test_ground_truths) in enumerate(test_loader):
            test_images = test_images.to(device)
            test_ground_truths = test_ground_truths.to(device)
            test_outputs = model(test_images)
            test_loss_values_over_batch = abs(test_outputs.squeeze(1) - test_ground_truths)
            for x in test_loss_values_over_batch:
                test_total_absolute_losses = test_total_absolute_losses + x
            test_total_counted += len(test_loss_values_over_batch)
            for index, limit in enumerate(config.loss_cutoff_for_accuracy):
                test_total_in_range[index] += len(test_loss_values_over_batch[test_loss_values_over_batch < limit])

        average_test_absolute_loss = test_total_absolute_losses / test_total_counted
        PATH = './saved_models/train_network' + str(epoch) + '.pth'
        torch.save(model.state_dict(), PATH)
        print(f"Test Loss {average_test_absolute_loss}")
        for index, limit in enumerate(config.loss_cutoff_for_accuracy):
            print("Test Accuracy within limit: " + str(limit) + ": " + str(test_total_in_range[index] / test_total_counted))
print('Finished Training')
