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
from loss import AvailableLaneLoss
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


loss_function = AvailableLaneLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=config.learning_rate)

batches_in_train = len(train_loader)
batches_in_test = len(test_loader)

for epoch in range(config.number_of_epochs):
    total_losses = 0.0
    train_images_counted = 0
    train_batches_counted = 0
    train_total_correct = 0
    train_left_correct = 0
    train_right_correct = 0
    for i, (images, train_ground_truths) in enumerate(train_loader):
        images = images.to(device)
        train_ground_truths = train_ground_truths.to(device)
        train_outputs = model(images)
        loss = loss_function(train_outputs, train_ground_truths)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        with torch.no_grad():
            train_predictions = torch.sigmoid(train_outputs)
            train_predictions = torch.round(train_predictions)
            train_ground_truths_of_zero = train_ground_truths.to(torch.long)
            train_ground_truths_of_zero = torch.where(train_ground_truths_of_zero == 1, 2, train_ground_truths_of_zero)
            train_ground_truths_of_zero = torch.where(train_ground_truths_of_zero == 0, 1, train_ground_truths_of_zero)
            train_ground_truths_of_zero = torch.where(train_ground_truths_of_zero == 2, 0, train_ground_truths_of_zero)
            train_predictions_of_zero = train_predictions.to(torch.long)
            train_predictions_of_zero = torch.where(train_predictions_of_zero == 1, 2, train_predictions_of_zero)
            train_predictions_of_zero = torch.where(train_predictions_of_zero == 0, 1, train_predictions_of_zero)
            train_predictions_of_zero = torch.where(train_predictions_of_zero == 2, 0, train_predictions_of_zero)
            # Metrics
            total_losses = total_losses + loss
            train_images_counted += train_ground_truths.shape[0]
            train_batches_counted += 1
            train_predictions = train_predictions.to(torch.long)
            train_ground_truths = train_ground_truths.to(torch.long)
            train_total_correct = train_total_correct + torch.count_nonzero(torch.bitwise_and(train_predictions, train_ground_truths)) + torch.count_nonzero(torch.bitwise_and(train_predictions_of_zero, train_ground_truths_of_zero))
            train_left_correct = train_left_correct + torch.count_nonzero(torch.bitwise_and(train_predictions[..., 0:1], train_ground_truths[..., 0:1])) + torch.count_nonzero(torch.bitwise_and(train_predictions_of_zero[..., 0:1], train_ground_truths_of_zero[..., 0:1]))
            train_right_correct = train_right_correct + torch.count_nonzero(torch.bitwise_and(train_predictions[..., 1:2], train_ground_truths[..., 1:2])) + torch.count_nonzero(torch.bitwise_and(train_predictions_of_zero[..., 1:2], train_ground_truths_of_zero[..., 1:2]))

    average_loss = total_losses / train_batches_counted
    print(f"Epoch: {epoch} Loss: {average_loss}")
    print(f"Training Accuracy: {train_total_correct / 2.0 / train_images_counted}")
    print(f"Training Right Accuracy: {train_left_correct / train_images_counted}")
    print(f"Training Right Accuracy: {train_right_correct / train_images_counted}")
    with torch.no_grad():
        right_lane_correct = 0
        left_lane_correct = 0
        test_total_losses = 0.0
        test_total_counted = 0
        test_batches_counted = 0
        test_total_correct = 0
        test_left_correct= 0
        test_right_correct = 0
        for i, (test_images, test_ground_truths) in enumerate(test_loader):
            test_images = test_images.to(device)
            test_ground_truths = test_ground_truths.to(device)
            test_outputs = model(test_images)
            test_predictions = torch.sigmoid(test_outputs)
            test_predictions = torch.round(test_predictions)
            test_ground_truths_of_zero =  test_ground_truths.to(torch.long)
            test_ground_truths_of_zero = torch.where(test_ground_truths_of_zero == 1, 2, test_ground_truths_of_zero)
            test_ground_truths_of_zero = torch.where(test_ground_truths_of_zero == 0, 1, test_ground_truths_of_zero)
            test_ground_truths_of_zero = torch.where(test_ground_truths_of_zero == 2, 0, test_ground_truths_of_zero)
            test_predictions_of_zero = test_predictions.to(torch.long)
            test_predictions_of_zero = torch.where(test_predictions_of_zero == 1, 2, test_predictions_of_zero)
            test_predictions_of_zero = torch.where(test_predictions_of_zero == 0, 1, test_predictions_of_zero)
            test_predictions_of_zero = torch.where(test_predictions_of_zero == 2, 0, test_predictions_of_zero)
            #metrics
            test_total_losses = test_total_losses + loss_function(test_outputs,test_ground_truths).item()
            test_total_counted = test_total_counted + test_outputs.shape[0]
            test_batches_counted += 1
            test_predictions = test_predictions.to(torch.long)
            test_ground_truths = test_ground_truths.to(torch.long)
            test_total_correct = test_total_correct + torch.count_nonzero(torch.bitwise_and(test_predictions,test_ground_truths)) + torch.count_nonzero(torch.bitwise_and(test_predictions_of_zero,test_ground_truths_of_zero))
            test_left_correct = test_left_correct + torch.count_nonzero(torch.bitwise_and(test_predictions[..., 0:1], test_ground_truths[..., 0:1])) + torch.count_nonzero(torch.bitwise_and(test_predictions_of_zero[..., 0:1], test_ground_truths_of_zero[..., 0:1]))
            test_right_correct = test_right_correct + torch.count_nonzero(torch.bitwise_and(test_predictions[..., 1:2], test_ground_truths[..., 1:2])) + torch.count_nonzero(torch.bitwise_and(test_predictions_of_zero[..., 1:2], test_ground_truths_of_zero[..., 1:2]))
        average_test_loss = test_total_losses / test_batches_counted
        print(f"Test Loss: {average_test_loss}")
        print(f"Test Accuracy: {test_total_correct / 2.0 / test_total_counted}")
        print(f"Test Left Accuracy: {test_left_correct / test_total_counted}")
        print(f"Test Right Accuracy: {test_right_correct / test_total_counted}")
        PATH = './saved_models/train_network' + str(epoch) + '.pth'
        torch.save(model.state_dict(), PATH)

print('Finished Training')
