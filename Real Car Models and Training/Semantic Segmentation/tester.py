import torch
from bayesian_model import BayesianModel
import matplotlib.pyplot as plt
import cv2 as cv
import config

device = "cuda" if torch.cuda.is_available() else "cpu"

model = BayesianModel()
model.load_state_dict(torch.load("saved_models/train_network.pth"))
model.to(device)
model.eval()

for idx in range(1, 200, 8):
    fig = plt.figure()
    for j in range(8):
        bgr_image = cv.imread('/home/ryan/TrainingData/real_car_semantic_segmentation/img/'+str(idx+1+j)+".png", cv.IMREAD_COLOR)
        unmodified_camera_image = cv.cvtColor(bgr_image, cv.COLOR_BGR2RGB)
        camera_image = cv.resize(unmodified_camera_image, (config.model_image_width, config.model_image_height))
        camera_tensor = torch.from_numpy(camera_image)
        camera_tensor = camera_tensor.permute(2, 0, 1)
        fig.add_subplot(8, 3, 1+3*j)
        plt.imshow(camera_tensor.permute(1, 2, 0))
        plt.axis('off')

        unmodified_label_image = cv.cvtColor(cv.imread('/home/ryan/TrainingData/real_car_semantic_segmentation/masks_machine/' + str(idx+1+j)+".png", cv.IMREAD_COLOR),cv.COLOR_BGR2GRAY)
        label_image = cv.resize(unmodified_label_image, (config.model_image_width, config.model_image_height),interpolation=cv.INTER_NEAREST)
        label_tensor = torch.from_numpy(label_image)
        #combine labels
        label_tensor = label_tensor.to(torch.long)
        label_tensor = torch.where(label_tensor == 4, 1, label_tensor)
        label_tensor = torch.where(label_tensor == 5, 2, label_tensor)
        label_tensor = torch.where(label_tensor == 6, 3, label_tensor)
        label_tensor =label_tensor.to(torch.uint8)
        fig.add_subplot(8, 3, 2+3*j)
        plt.imshow(label_tensor)
        plt.axis('off')
        camera_tensor = camera_tensor.unsqueeze(0).to(torch.float32).to(device)
        prediction = model(camera_tensor)
        prediction = prediction.argmax(1).squeeze().to("cpu")
        fig.add_subplot(8, 3, 3+3*j)
        plt.imshow(prediction)
        plt.axis('off')
    breakSpot = 1