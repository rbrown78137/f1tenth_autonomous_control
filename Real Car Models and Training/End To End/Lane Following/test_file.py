import torch
import torchvision
from model import Model
from torchsummary import summary
model = Model()
model.load_state_dict(torch.load("saved_models/train_network.pth"))
model.eval()
model.to("cuda")

summary(model,( 3, 256, 256))
test = 1