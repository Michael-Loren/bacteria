import sys
import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models

# Add the directory to sys.path
module_path = os.path.abspath(os.path.join('transform-based-layers', 'layers'))
sys.path.append(module_path)

from WHT import WHTConv2D
from DCT import DCTConv2D
from BWT import BWTConv2D
from DChT import DChTConv2D
class CustomResNet50(nn.Module):
    def __init__(self, num_classes=10, transform_layer="DCT", height=224, width=224, in_channels=31, out_channels=3, pods=3, residual=False):
        super(CustomResNet50, self).__init__()
        # Load the pretrained ResNet-50 model
        self.resnet50 = models.resnet50(weights=models.ResNet50_Weights.IMAGENET1K_V1)
        
        # Freeze the pretrained layers
        for param in self.resnet50.parameters():
            param.requires_grad = False

        # Dynamically select the transform layer
        if transform_layer == "WHT":
            self.t_layer = WHTConv2D(height=height, width=width, in_channels=in_channels, out_channels=out_channels, pods=pods, residual=residual)
        elif transform_layer == "DCT":
            self.t_layer = DCTConv2D(height=height, width=width, in_channels=in_channels, out_channels=out_channels, pods=pods, residual=residual)
        elif transform_layer == "BWT":
            self.t_layer = BWTConv2D(height=height, width=width, in_channels=in_channels, out_channels=out_channels, pods=pods, residual=residual)
        elif transform_layer == "DChT":
            self.t_layer = DChTConv2D(height=height, width=width, in_channels=in_channels, out_channels=out_channels, pods=pods, residual=residual)
        elif transform_layer == "Conv1x1":
            self.t_layer = nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=1, stride=1, padding=0)
        else:
            raise ValueError(f"Unknown transform layer: {transform_layer}")
        
        # Modify the final fully connected layer to match the number of classes
        num_features = self.resnet50.fc.in_features
        self.resnet50.fc = nn.Linear(num_features, num_classes)
        
        # Ensure the new fully connected layer has requires_grad=True
        for param in self.resnet50.fc.parameters():
            param.requires_grad = True

    def forward(self, x):
        # Pass the input through the transform layer
        x = self.t_layer(x)
        # Then pass it through the ResNet model
        x = self.resnet50(x)
        return x

