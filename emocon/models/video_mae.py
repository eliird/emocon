import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models
from .model_utils import load_mae_model

class Encoder(nn.Module):
    def __init__(self, out_dim=64):
        super(Encoder, self).__init__()
        self.conv1 = nn.Conv2d(3, 16, kernel_size=3, stride=1, padding=1)
        self.conv2 = nn.Conv2d(16, 32, kernel_size=3, stride=1, padding=1)
        self.conv3 = nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1)
        self.conv4 = nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1)
        self.pool = nn.MaxPool2d(2, 2)

        # projection MLP
        self.l1 = nn.Linear(64, 64)
        self.l2 = nn.Linear(64, out_dim)

    def forward(self, x):
        x = self.conv1(x)
        x = F.relu(x)
        x = self.pool(x)

        x = self.conv2(x)
        x = F.relu(x)
        x = self.pool(x)

        x = self.conv3(x)
        x = F.relu(x)
        x = self.pool(x)

        x = self.conv4(x)
        x = F.relu(x)
        x = self.pool(x)

        h = torch.mean(x, dim=[2, 3])

        x = self.l1(h)
        x = F.relu(x)
        x = self.l2(x)

        return h, x


class VideoMAESimCLR(nn.Module):
    def __init__(self, out_dim=256, model_name="MCG-NJU/videomae-base"):
        v_mae = load_mae_model(model_name)
        num_ftrs = v_mae.classifier.in_features
        
        self.features = nn.Sequential(*list(v_mae.children())[:-1])
        self.l1 = nn.Linear(num_ftrs, num_ftrs)
        self.l2 = nn.Linear(num_ftrs, out_dim)
        
    def forward(self, x):
        h = self.features(x)
        h = h.squeeze()

        x = self.l1(h)
        x = F.relu(x)
        x = self.l2(x)
        return h, x