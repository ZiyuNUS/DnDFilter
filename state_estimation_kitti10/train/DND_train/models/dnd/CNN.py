import torch
import torch.nn as nn
import torch.nn.functional as F

class SensorModel(nn.Module):
    def __init__(self):
        super(SensorModel, self).__init__()

        self.conv1 = nn.Conv2d(
            in_channels=6, out_channels=16, kernel_size=7, stride=1, padding=3
        )
        self.layer_norm1 = nn.LayerNorm([16, 50, 150])

        self.conv2 = nn.Conv2d(
            in_channels=16, out_channels=16, kernel_size=5, stride=(1, 2), padding=2
        )
        self.layer_norm2 = nn.LayerNorm([16, 50, 75])

        self.conv3 = nn.Conv2d(
            in_channels=16, out_channels=16, kernel_size=5, stride=(1, 2), padding=(2,1)
        )
        self.layer_norm3 = nn.LayerNorm([16, 50, 37])

        self.conv4 = nn.Conv2d(
            in_channels=16, out_channels=16, kernel_size=5, stride=(2, 2), padding=(2,1)
        )
        self.layer_norm4 = nn.LayerNorm([16, 25, 18])

        self.dropout = nn.Dropout(p=0.3)

        # Compute the flattened size after conv layers
        self.flattened_size = 16 * 25 * 18

        # Fully connected layers
        self.fc1 = nn.Linear(in_features=self.flattened_size, out_features=128)
        self.fc2 = nn.Linear(in_features=128, out_features=128)

    def forward(self, x):
        # Convolutional layers with ReLU activation and layer normalization
        x = F.relu(self.conv1(x))
        x = self.layer_norm1(x)

        x = F.relu(self.conv2(x))
        x = self.layer_norm2(x)

        x = F.relu(self.conv3(x))
        x = self.layer_norm3(x)

        x = F.relu(self.conv4(x))
        x = self.layer_norm4(x)

        x = self.dropout(x)

        # Flatten the tensor before feeding into fully connected layers
        x = torch.flatten(x, start_dim=1)

        # Fully connected layers with ReLU activation
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))

        return x
