# hyperspectral_dataset.py

import os
import scipy.io
import torch
from torch.utils.data import Dataset
import torchvision.transforms as transforms

# Custom Transform Classes
class NormalizeProfile:
    def __init__(self, mean, std):
        self.mean = torch.tensor(mean)
        self.std = torch.tensor(std)

    def __call__(self, profile):
        return (profile - self.mean) / self.std

class NormalizeCube:
    def __init__(self, mean, std):
        self.mean = torch.tensor(mean)
        self.std = torch.tensor(std)

    def __call__(self, cube):
        return (cube - self.mean[:, None, None]) / self.std[:, None, None]

# Custom Dataset Class
class HyperspectralDataset(Dataset):
    def __init__(self, root_dir, cube_transform=None, profile_transform=None):
        self.root_dir = root_dir
        self.data = []
        self.labels = []
        self.label_map = {}
        self.cube_transform = cube_transform
        self.profile_transform = profile_transform

        species_dirs = os.listdir(root_dir)
        for idx, species in enumerate(species_dirs):
            species_dir = os.path.join(root_dir, species)
            if os.path.isdir(species_dir):
                self.label_map[species] = idx
                mat_files = [f for f in os.listdir(species_dir) if f.endswith('.mat')]
                for mat_file in mat_files:
                    file_path = os.path.join(species_dir, mat_file)
                    self.data.append(file_path)
                    self.labels.append(idx)
        
        print("Label mapping:", self.label_map)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        mat_file = self.data[idx]
        label = self.labels[idx]

        # Load .mat file
        mat_contents = scipy.io.loadmat(mat_file)
        cube = mat_contents['cube']
        profile = mat_contents['profile']

        # Convert to PyTorch tensor and permute dimensions
        cube = torch.tensor(cube, dtype=torch.float32).permute(2, 0, 1)
        profile = torch.tensor(profile, dtype=torch.float32)
        label = torch.tensor(label, dtype=torch.long)

        if self.cube_transform:
            cube = self.cube_transform(cube)
        
        if self.profile_transform:
            profile = self.profile_transform(profile)

        return cube, profile, label

    def get_image_path(self, idx):
        return self.data[idx]

    def set_transforms(self, cube_transform=None, profile_transform=None):
        self.cube_transform = cube_transform
        self.profile_transform = profile_transform

# Function to compute statistics
def compute_statistics(dataset):
    all_profiles = []
    all_cubes = []

    for i in range(len(dataset)):
        mat_file = dataset.get_image_path(i)
        mat_contents = scipy.io.loadmat(mat_file)
        cube = mat_contents['cube']
        profile = mat_contents['profile']

        cube = torch.tensor(cube, dtype=torch.float32).permute(2, 0, 1)
        profile = torch.tensor(profile, dtype=torch.float32)

        all_cubes.append(cube)
        all_profiles.append(profile)

    all_cubes_tensor = torch.stack(all_cubes)
    all_profiles_tensor = torch.stack(all_profiles)

    channel_means_cube = all_cubes_tensor.mean(dim=[0, 2, 3])
    channel_stds_cube = all_cubes_tensor.std(dim=[0, 2, 3])

    channel_means_profile = all_profiles_tensor.mean(dim=0)
    channel_stds_profile = all_profiles_tensor.std(dim=0)

    return channel_means_cube, channel_stds_cube, channel_means_profile, channel_stds_profile

