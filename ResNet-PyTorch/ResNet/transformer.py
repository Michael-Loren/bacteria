import sys
import os
import torch
import torch.nn as nn
import torchvision.models as models
from transformers import AutoModelForImageClassification

# Add the directory to sys.path
module_path = os.path.abspath(os.path.join('..', '..', '..', 'transform-based-layers', 'layers'))
sys.path.append(module_path)

# Import custom transform layers
from WHT import WHTConv2D
from DCT import DCTConv2D
from BWT import BWTConv2D
from DChT import DChTConv2D

class TransformerMod(nn.Module):
    def __init__(self, num_classes=0, transformer_model="facebook/deit-base-patch16-224", transform_layer="DCT", height=224, width=224, in_channels=31, out_channels=3, pods=3, residual=False):
        super(TransformerMod, self).__init__()
        
        # Load the specified pretrained model
        self.customtransformer = AutoModelForImageClassification.from_pretrained(transformer_model)
        
        # Freeze the pretrained layers
        for param in self.customtransformer.parameters():
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
        
        # Ensure the final classifier has requires_grad=True
        self.customtransformer.classifier.requires_grad_(True)

        # Modify the final classifier layer to match the number of classes
        num_features = self.customtransformer.classifier.in_features
        self.customtransformer.classifier = nn.Linear(num_features, num_classes)
        
    def forward(self, x):
        # Pass the input through the transform layer
        x = self.t_layer(x)
        # Pass the transformed input through the transformer model
        x = self.customtransformer(x)
        return x

