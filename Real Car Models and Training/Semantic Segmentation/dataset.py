import torch
from torch.utils.data import Dataset
import cv2 as cv
import os
import matplotlib.pyplot as plt
import config


class CustomImageDataset(Dataset):
    def __init__(self, transform=None, target_transform=None):

        self.transform = transform
        self.target_transform = target_transform
        self.number_of_images = len([name for name in os.listdir('/home/ryan/TrainingData/real_car_semantic_segmentation/masks_machine')])
        self.labels = torch.zeros([self.number_of_images, config.model_image_width, config.model_image_height], dtype=torch.long)
        self.images = torch.zeros([self.number_of_images, 3, config.model_image_width, config.model_image_height])

        for idx in range(self.number_of_images):
            bgr_image = cv.imread('/home/ryan/TrainingData/real_car_semantic_segmentation/img/'+str(idx+1)+".png", cv.IMREAD_COLOR)
            unmodified_camera_image = cv.cvtColor(bgr_image, cv.COLOR_BGR2RGB)
            camera_image = cv.resize(unmodified_camera_image, (config.model_image_width, config.model_image_height))
            camera_tensor = torch.from_numpy(camera_image)
            camera_tensor = camera_tensor.permute(2, 0, 1)
            self.images[idx] = camera_tensor
            # plt.imshow(camera_tensor.permute(1, 2, 0))
            # plt.show()
            unmodified_label_image = cv.cvtColor(cv.imread('/home/ryan/TrainingData/real_car_semantic_segmentation/masks_machine/' + str(idx+1)+".png", cv.IMREAD_COLOR),cv.COLOR_BGR2GRAY)
            label_image = cv.resize(unmodified_label_image, (config.model_image_width, config.model_image_height),interpolation=cv.INTER_NEAREST)
            label_tensor = torch.from_numpy(label_image)
            #combine labels
            label_tensor = label_tensor.to(torch.long)
            label_tensor = torch.where(label_tensor == 4, 1, label_tensor)
            label_tensor = torch.where(label_tensor == 5, 2, label_tensor)
            label_tensor = torch.where(label_tensor == 6, 3, label_tensor)
            label_tensor =label_tensor.to(torch.uint8)
            self.labels[idx] = label_tensor
            # plt.imshow(label_tensor)
            # plt.show()
            print(f"Loaded image {idx+1} / {self.number_of_images}")

    def __len__(self):
        return self.number_of_images

    def __getitem__(self, idx):
        image = self.images[idx]
        label = self.labels[idx]
        return image,label
