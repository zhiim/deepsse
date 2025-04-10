from torch import nn

from model.base_model import BaseModel


class DOALowSNRNet(BaseModel):
    def __init__(self, num_out_grids=180):
        """CNN based DOA estimation network

        Args:
            num_out_grids (int): number of angle girds used

        References:
            Papageorgiou, Georgios K., Mathini Sellathurai, and Yonina C. Eldar.
            “Deep Networks for Direction-of-Arrival Estimation in Low SNR.”
            IEEE Transactions on Signal Processing 69 (2021): 3714-29.
            https://doi.org/10.1109/TSP.2021.3089927.
        """
        super().__init__()

        self.conv_layer1 = nn.Sequential(
            nn.Conv2d(
                in_channels=3,
                out_channels=256,
                kernel_size=(3, 3),
                stride=2,
                padding=0,
            ),
            nn.BatchNorm2d(256),
            nn.ReLU(),
        )
        self.conv_layer2 = nn.Sequential(
            nn.Conv2d(
                in_channels=256,
                out_channels=256,
                kernel_size=(2, 2),
                stride=1,
                padding=0,
            ),
            nn.BatchNorm2d(256),
            nn.ReLU(),
        )
        self.conv_layer3 = nn.Sequential(
            nn.Conv2d(
                in_channels=256,
                out_channels=256,
                kernel_size=(2, 2),
                stride=1,
                padding=0,
            ),
            nn.BatchNorm2d(256),
            nn.ReLU(),
        )
        self.conv_layer4 = nn.Sequential(
            nn.Conv2d(
                in_channels=256,
                out_channels=256,
                kernel_size=(2, 2),
                stride=1,
                padding=0,
            ),
            nn.BatchNorm2d(256),
            nn.ReLU(),
        )

        self.flatten = nn.Flatten()

        self.fc1 = nn.Sequential(nn.LazyLinear(out_features=4096), nn.ReLU())
        self.fc2 = nn.Sequential(
            nn.Linear(in_features=4096, out_features=2048), nn.ReLU()
        )
        self.fc3 = nn.Sequential(
            nn.Linear(in_features=2048, out_features=1024), nn.ReLU()
        )
        self.fc4 = nn.Sequential(
            nn.Linear(in_features=1024, out_features=num_out_grids),
            nn.Sigmoid(),
        )

    def forward(self, x):
        x = self.conv_layer1(x)
        x = self.conv_layer2(x)
        x = self.conv_layer3(x)
        x = self.conv_layer4(x)

        x = self.flatten(x)

        x = self.fc1(x)
        x = self.fc2(x)
        x = self.fc3(x)
        x = self.fc4(x)

        return x
