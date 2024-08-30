import sys
import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models

# Add the directory to sys.path
module_path = os.path.abspath(os.path.join('..', '..', '..', 'transform-based-layers', 'layers'))
sys.path.append(module_path)

# Now you can import the WHTConv2D class
from WHT import WHTConv2D
from DCT import DCTConv2D
from BWT import BWTConv2D
from DChT import DChTConv2D
class CustomResNet50(nn.Module):
    def __init__(self, num_classes=10):
        super(CustomResNet50, self).__init__()
        # Load the pretrained ResNet-50 model
        self.resnet50 = models.resnet50(weights=models.ResNet50_Weights.IMAGENET1K_V1)
        
        # Freeze the pretrained layers
        for param in self.resnet50.parameters():
            param.requires_grad = False

	# Add a new 1x1 convolutional layer to reduce channels from 31 to 3
        # self.conv1x1 = nn.Conv2d(in_channels=31, out_channels=3, kernel_size=1, stride=1, padding=0)
        
        # Add a new WHT layer
        # self.t_layer = WHTConv2D(height=256, width=256, in_channels=31, out_channels=3, pods=3, residual=False)
        self.t_layer = DCTConv2D(height=256, width=256, in_channels=31, out_channels=3, pods=3, residual=False)
        # self.t_layer = BWTConv2D(height=256, width=256, in_channels=31, out_channels=3, pods=3, residual=False)
        #self.t_layer = DChTConv2D(height=256, width=256, in_channels=31, out_channels=3, pods=3, residual=False)
        num_features = self.resnet50.fc.in_features
        self.resnet50.fc = nn.Linear(num_features, num_classes)
        
        # Ensure the new fully connected layer has requires_grad=True
        for param in self.resnet50.fc.parameters():
            param.requires_grad = True

    def forward(self, x):
        # x = self.conv1x1(x)
        # Pass the input through the new WHT layer
        x = self.t_layer(x)
        # Then pass it through the ResNet model
        x = self.resnet50(x)
        return x



