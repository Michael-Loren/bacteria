import torch
import torch.nn as nn
from transformer import TransformerMod
from pnet import pNet

# Define paths to the checkpoint files
transformer_checkpoint_path = 'resnet.pth'
pnet_checkpoint_path = 'pnet.pth'

# Load the pretrained weights
transformer_checkpoint = torch.load(transformer_checkpoint_path)
pnet_checkpoint = torch.load(pnet_checkpoint_path)


# Initialize the transformer model BEFORE popping
transformer_mod_pre = TransformerMod()

print("\n=== Classifier architecture before popping ===")
print(transformer_mod_pre.customtransformer.classifier)

print("\n=== Classifier parameters before popping ===")
for name, param in transformer_mod_pre.customtransformer.classifier.named_parameters():
    print(f"{name}: shape={tuple(param.shape)}")


print("Transformer keys before pop:", 
      [k for k in transformer_checkpoint.keys() if "classifier" in k])
# Remove the classifier weights from the TransformerMod checkpoint
transformer_checkpoint.pop('customtransformer.classifier.weight', None)
transformer_checkpoint.pop('customtransformer.classifier.bias', None)

# Inspect keys after popping
print("Transformer keys after pop:", 
      [k for k in transformer_checkpoint.keys() if "classifier" in k])
      
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

# Freeze most of the pretrained layers, unfreezing selected ones for fine-tuning
for name, param in transformer_mod.named_parameters():
    # Unfreeze the last transformer block (adjust as needed for specific layers)
    if 'layer4' in name:
        param.requires_grad = True
    else:
        param.requires_grad = False

for param in pnet_model.parameters():
    param.requires_grad = False
pnet_model.dense1.requires_grad = True  # Example of unfreezing a layer in pNet

class CombinedModel(nn.Module):
    def __init__(self, num_classes=10):
        super(CombinedModel, self).__init__()
        self.transformer_mod = transformer_mod
        self.pnet = pnet_model

        # Intermediate layer to help blend and reduce dimensions after concatenation
        fdim = 256
        self.intermediate_layer = nn.Linear(768 + 100, fdim)
        self.fc = nn.Linear(fdim, num_classes)
        self.layer_norm = nn.LayerNorm(fdim)  # Normalize after the intermediate layer
        
    def forward(self, cube, profile):
        # Get the output logits directly from transformer_mod
        transformer_output = self.transformer_mod(cube)
        transformer_out = transformer_output.logits if hasattr(transformer_output, 'logits') else transformer_output
        
        pnet_out = self.pnet(profile)
        
        # Concatenate the outputs from both models
        combined = torch.cat((transformer_out, pnet_out), dim=1)
        
        # Intermediate dense layer to blend features and add non-linearity
        combined = self.intermediate_layer(combined)
        combined = self.layer_norm(combined)
        combined = nn.ReLU()(combined)
        
        # Final output layer for classification
        out = self.fc(combined)
        
        return out
