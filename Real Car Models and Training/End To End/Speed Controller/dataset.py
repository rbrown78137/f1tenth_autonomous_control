import torch
import torchvision
from torch.utils.data import Dataset
import matplotlib.pyplot as plt
from pathlib import Path
import struct
import os
import numpy as np
import time
import cv2 as cv
import config

device = "cuda" if torch.cuda.is_available() else "cpu"


class CustomImageDataset(Dataset):
    def __init__(self, transform=None, target_transform=None):
        self.transform = transform
        self.target_transform = target_transform
        self.data = Path((config.dataset_path + "/" + config.dataset_input_file)).read_bytes()
        self.labels = torch.zeros([self.__len__()], dtype=torch.float)
        self.images = torch.zeros([self.__len__(),config.input_dimensions, config.input_image_width, config.input_image_height], dtype=torch.float)
        for idx in range(self.__len__()):
            unmodified_camera_image = cv.imread(config.dataset_path + "/" + config.dataset_image_folder_name + "/" + str(idx) + ".png", cv.IMREAD_COLOR)
            rgb_camera_image = cv.cvtColor(unmodified_camera_image, cv.COLOR_BGR2RGB)
            # or cv.IMREAD_GRAYSCALE
            resized_image = cv.resize(rgb_camera_image, (config.input_image_width, config.input_image_height))
            camera_tensor = torch.from_numpy(resized_image)
            camera_tensor = camera_tensor.permute(2, 0, 1)

            self.images[idx] = camera_tensor
            self.labels[idx] = struct.unpack('d', self.data[config.input_byte_size*idx:config.input_byte_size*(idx + 1)])[0]
            print(f"Loaded image {idx}")

    def __len__(self):
        return int(self.data.__len__() / config.input_byte_size)

    def __getitem__(self, idx):
        image = self.images[idx]
        output = self.labels[idx]
        return image, output
