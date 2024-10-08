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

# Custom model class
class TransformerMod(nn.Module):
    def __init__(self, num_classes=10):
        super(TransformerMod, self).__init__()
        # Load the pretrained model
        self.customtransformer = AutoModelForImageClassification.from_pretrained("facebook/deit-base-patch16-224")
        
        # Freeze the pretrained layers
        for param in self.customtransformer.parameters():
            param.requires_grad = False

        h = 224
        w = 224
        # Add a new transform layer
        #self.t_layer = WHTConv2D(height=h, width=w, in_channels=31, out_channels=3, pods=3, residual=False)
        self.t_layer = DCTConv2D(height=h, width=w, in_channels=31, out_channels=3, pods=3, residual=False)
        #self.t_layer = BWTConv2D(height=h, width=w, in_channels=31, out_channels=3, pods=3, residual=False)
        #self.t_layer = DChTConv2D(height=h, width=w, in_channels=31, out_channels=3, pods=3, residual=False)
        
        # Ensure the final classifier has requires_grad=True
        self.customtransformer.classifier.requires_grad_(True)

        # Modify the final classifier layer to match the number of classes
        num_features = self.customtransformer.classifier.in_features
        self.customtransformer.classifier = nn.Linear(num_features, num_classes)
        
    def forward(self, x):
        # Pass the input through the transform layer
        x = self.t_layer(x)
        # Pass the transformed input through the EfficientNet model
        x = self.customtransformer(x)
        return x


