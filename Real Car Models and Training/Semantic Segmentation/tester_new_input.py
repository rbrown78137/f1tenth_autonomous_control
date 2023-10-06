import torch
from bayesian_model import BayesianModel
import matplotlib.pyplot as plt
import cv2 as cv
import config

device = "cuda" if torch.cuda.is_available() else "cpu"

model = BayesianModel()
model.load_state_dict(torch.load("saved_models/train_network.pth"))
# model = torch.jit.load("saved_models/converted_jit_model.pt")
model.to(device)
model.eval()

for idx in range(1130, 1430, 8):
    fig = plt.figure()
    for j in range(8):
        bgr_image = cv.imread('/home/ryan/TrainingData/real_car_photos/'+str(idx+1+j)+".png", cv.IMREAD_COLOR)
        unmodified_camera_image = cv.cvtColor(bgr_image, cv.COLOR_BGR2RGB)
        camera_image = cv.resize(unmodified_camera_image, (config.model_image_width, config.model_image_height))
        camera_tensor = torch.from_numpy(camera_image)
        camera_tensor = camera_tensor.permute(2, 0, 1)
        fig.add_subplot(8, 2, 1+2*j)
        plt.imshow(camera_tensor.permute(1, 2, 0))
        plt.axis('off')

        camera_tensor = camera_tensor.unsqueeze(0).to(torch.float32).to(device)
        prediction = model(camera_tensor)
        prediction = prediction.argmax(1).squeeze().to("cpu")
        fig.add_subplot(8, 2, 2+2*j)
        plt.imshow(prediction)
        plt.axis('off')
    breakSpot = 1