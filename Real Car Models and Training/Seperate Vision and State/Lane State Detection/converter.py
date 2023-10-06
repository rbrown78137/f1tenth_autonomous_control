import torch
import torchvision

import config
from model import Model

model = Model()
model.load_state_dict(torch.load("saved_models/train_network.pth"))
model.eval()

example = torch.rand(1, config.input_channels, 256, 256)

traced_script_module = torch.jit.trace(model, example)
traced_script_module.save("saved_models/lane_state_network.pt")

print("Done")
