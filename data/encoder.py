import torch
import torch.nn as nn
import torch.nn.functional as F


# Define the encoder model
class SphereEncoder(nn.Module):
    def __init__(self):
        super(SphereEncoder, self).__init__()
        self.f = nn.Sequential(
            nn.Linear(3, 30),
            nn.LeakyReLU(negative_slope=0.01),
            nn.Linear(30, 150),
            nn.LeakyReLU(negative_slope=0.01),
            nn.Linear(150, 150),
            nn.LeakyReLU(negative_slope=0.01),
            nn.Linear(150, 150),
            nn.LeakyReLU(negative_slope=0.01),
            nn.Linear(150, 150),
            nn.LeakyReLU(negative_slope=0.01),
            nn.Linear(150, 30),
            nn.LeakyReLU(negative_slope=0.01),
            nn.Linear(30, 3)
        )

    def forward(self, x):
        return F.normalize(self.f(x), dim=1, p=2)

