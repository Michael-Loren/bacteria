import os
from PIL import Image
import torch
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as transforms

# Custom Dataset Class for RGB Images
class RGBDataset(Dataset):
    def __init__(self, root_dir, transform=None):
        self.root_dir = root_dir
        self.data = []
        self.labels = []
        self.label_map = {}
        self.transform = transform

        species_dirs = os.listdir(root_dir)
        for idx, species in enumerate(species_dirs):
            species_dir = os.path.join(root_dir, species)
            if os.path.isdir(species_dir):
                self.label_map[species] = idx
                img_files = [f for f in os.listdir(species_dir) if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
                for img_file in img_files:
                    file_path = os.path.join(species_dir, img_file)
                    self.data.append(file_path)
                    self.labels.append(idx)
        
        print("Label mapping:", self.label_map)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        img_file = self.data[idx]
        label = self.labels[idx]

        # Load image
        image = Image.open(img_file).convert('RGB')

        if self.transform:
            image = self.transform(image)

        label = torch.tensor(label, dtype=torch.long)

        return image, label

    def get_image_path(self, idx):
        return self.data[idx]

    def set_transform(self, transform=None):
        self.transform = transform

