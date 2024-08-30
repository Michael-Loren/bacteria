import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models
from ResNet import CustomResNet50  
from pnet import pNet

# Define paths to the checkpoint files
resnet_checkpoint_path = 'resnet.pth'
pnet_checkpoint_path = 'pnet.pth'

# Load the pretrained weights
resnet_checkpoint = torch.load(resnet_checkpoint_path)
pnet_checkpoint = torch.load(pnet_checkpoint_path)

# Remove the fully connected layer weights from the checkpoint
resnet_checkpoint.pop('resnet50.fc.weight', None)
resnet_checkpoint.pop('resnet50.fc.bias', None)

pnet_checkpoint.pop('dense2.weight', None)
pnet_checkpoint.pop('dense2.bias', None)

# Initialize the models
custom_resnet50 = CustomResNet50()
pnet_model = pNet()

# Load the weights into the models
custom_resnet50.load_state_dict(resnet_checkpoint, strict=False)
pnet_model.load_state_dict(pnet_checkpoint, strict=False)

# Remove the fully connected layer from CustomResNet50
custom_resnet50.resnet50.fc = nn.Identity()

# Remove dropout2 and dense2 layers from pNet
pnet_model.dropout2 = nn.Identity()
pnet_model.dense2 = nn.Identity()

# Freeze the pretrained layers
for param in custom_resnet50.parameters():
    param.requires_grad = False

for param in pnet_model.parameters():
    param.requires_grad = False

class CombinedModel(nn.Module):
    def __init__(self, num_classes=10):
        super(CombinedModel, self).__init__()
        self.custom_resnet50 = custom_resnet50
        self.pnet = pnet_model
        
        # Ensure the output layer is as per your custom resnet and pnet
        # Assuming output of CustomResNet50 is 2048 and pNet output is 100
        self.fc = nn.Linear(2048 + 100, num_classes)
    
    def forward(self, cube, profile):
        resnet_out = self.custom_resnet50(cube)
        pnet_out = self.pnet(profile)
        
        # Concatenate the outputs
        combined = torch.cat((resnet_out, pnet_out), dim=1)
        
        # Pass through the final dense layer
        out = self.fc(combined)
        
        return out

