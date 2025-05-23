import torch
import torch.nn as nn
import torchvision.models as models


class ResNet50Encoder(nn.Module):
    def __init__(self):
        super(ResNet50Encoder, self).__init__()
        # Load ResNet-50
        resnet = models.resnet50()
        
        # Remove the final classification layer
        self.encoder = nn.Sequential(*list(resnet.children())[:-1])
        
    def forward(self, x):
        # ResNet-50 expects input of shape [B, 3, 224, 224]
        x = self.encoder(x)
        # Remove the extra dimension added by the avg pool
        x = x.squeeze(-1).squeeze(-1)
        return x


class Expander(nn.Module):
    def __init__(self, input_dim=2048, hidden_dim=8192, output_dim=8192):
        super(Expander, self).__init__()
        
        self.expander = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, output_dim),
        )
        
    def forward(self, x):
        return self.expander(x)


def create_vicreg_components():
    """
    Helper function to create both the encoder and expander components.
    
    Args:
        pretrained (bool): Whether to use pretrained weights for ResNet-50
        
    Returns:
        tuple: (encoder, expander) both ready to use
    """
    encoder = ResNet50Encoder()
    expander = Expander()
    return encoder, expander
