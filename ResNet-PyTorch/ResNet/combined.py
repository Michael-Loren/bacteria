import torch
import torch.nn as nn
from transformer import TransformerMod
from pnet import pNet

# Define paths to the checkpoint files
transformer_checkpoint_path = 'transformer.pth'
pnet_checkpoint_path = 'pnet.pth'

# Load the pretrained weights
transformer_checkpoint = torch.load(transformer_checkpoint_path)
pnet_checkpoint = torch.load(pnet_checkpoint_path)

# Remove the classifier weights from the TransformerMod checkpoint
transformer_checkpoint.pop('customtransformer.classifier.weight', None)
transformer_checkpoint.pop('customtransformer.classifier.bias', None)

# Remove the fully connected layer weights from the pNet checkpoint
pnet_checkpoint.pop('dense2.weight', None)
pnet_checkpoint.pop('dense2.bias', None)

# Initialize the models
transformer_mod = TransformerMod()
pnet_model = pNet()

# Load the weights into the models
transformer_mod.load_state_dict(transformer_checkpoint, strict=False)
pnet_model.load_state_dict(pnet_checkpoint, strict=False)

# Remove the final layers to use as feature extractors
transformer_mod.customtransformer.classifier = nn.Identity()
pnet_model.dense2 = nn.Identity()
pnet_model.dropout2 = nn.Identity()

# Freeze the pretrained layers
for param in transformer_mod.parameters():
    param.requires_grad = False

for param in pnet_model.parameters():
    param.requires_grad = False

class CombinedModel(nn.Module):
    def __init__(self, num_classes=10):
        super(CombinedModel, self).__init__()
        self.transformer_mod = transformer_mod
        self.pnet = pnet_model
        
        # Adjust the fully connected layer based on the output dimensions of TransformerMod and pNet
        # Assuming the output of TransformerMod is 768 and pNet is 100 (adjust if needed)
        self.fc = nn.Linear(768 + 100, num_classes)
    
    def forward(self, cube, profile):
        # Get the output logits directly from transformer_mod
        transformer_output = self.transformer_mod(cube)
        transformer_out = transformer_output.logits if hasattr(transformer_output, 'logits') else transformer_output
        
        pnet_out = self.pnet(profile)
        
        # Concatenate the outputs from both models
        combined = torch.cat((transformer_out, pnet_out), dim=1)
        
        # Pass through the final dense layer for classification
        out = self.fc(combined)
        
        return out
