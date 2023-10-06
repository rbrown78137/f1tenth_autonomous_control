from torch import nn
import config


class Model(nn.Module):
    def __init__(self):
        super(Model, self).__init__()
        self.conv_layer = nn.Sequential(
            nn.AvgPool2d(2),
            # 128 128
            nn.Conv2d(config.input_dimensions, 24, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(24, 24, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.AvgPool2d(2),
            nn.AvgPool2d(2),
            # 32 32
            nn.Conv2d(24, 36, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(36, 36, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.AvgPool2d(2),
            # 16 16
            nn.Conv2d(36, 48, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(48, 48, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.AvgPool2d(2),
            # 8 8
            nn.Conv2d(48, 64, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 64, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.AvgPool2d(2),
            # 4 4
            nn.Conv2d(64, 64, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 64, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Flatten(),
        )
        self.linear_layers = nn.Sequential(
            nn.Linear(in_features=64 * 4 * 4, out_features=100),
            nn.ReLU(),
            nn.Linear(in_features=100, out_features=50),
            nn.ReLU(),
            nn.Linear(in_features=50, out_features=10),
            nn.ReLU(),
            nn.Linear(in_features=10, out_features=1)
        )

    def forward(self, image):
        output = self.conv_layer(image)
        output = self.linear_layers(output)
        return output
