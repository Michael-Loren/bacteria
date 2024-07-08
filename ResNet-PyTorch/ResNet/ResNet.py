import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models


class CustomResNet50(nn.Module):
    def __init__(self, num_classes=10):
        super(CustomResNet50, self).__init__()
        # Load the pretrained ResNet-50 model
        self.resnet50 = models.resnet50(weights=models.ResNet50_Weights.IMAGENET1K_V1)
        
	# Freeze the pretrained layers
        for param in self.resnet50.parameters():
            param.requires_grad = False

        # Add a new convolutional layer
        self.conv1x1 = nn.Conv2d(in_channels=31, out_channels=3, kernel_size=1, stride=1, padding=0)
        	
	# Ensure the new convolutional layer has requires_grad=True
        for param in self.conv1x1.parameters():
            param.requires_grad = True        	
        	

        #self.resnet50.fc = nn.Identity() # drop the last layer for combining later
        num_features = self.resnet50.fc.in_features
        self.resnet50.fc = nn.Linear(num_features, num_classes)
        
        # Ensure the new fully connected layer has requires_grad=True
        for param in self.resnet50.fc.parameters():
            param.requires_grad = True

    def forward(self, x):
        # Pass the input through the new 1x1 convolutional layer
        x = self.conv1x1(x)
        # Then pass it through the ResNet model
        x = self.resnet50(x)
        return x


