import os
import scipy.io
import torch
from torch.utils.data import Dataset
import torchvision.transforms as transforms

# Custom Dataset Class for Hyperspectral Data
class HyperspectralDataset(Dataset):
    def __init__(self, root_dir, transform=None):
        self.root_dir = root_dir
        self.data = []
        self.labels = []
        self.label_map = {}
        self.transform = transform

        # Populate dataset with file paths and labels
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
        cube = torch.tensor(cube, dtype=torch.float32).permute(2, 0, 1)  # Channels first: (31, H, W)
        profile = torch.tensor(profile, dtype=torch.float32)
        label = torch.tensor(label, dtype=torch.long)

        # Combine cube and profile for transformation (if needed)
        # If specific transforms for cube or profile are required, handle them in the transform function
        sample = {'cube': cube, 'profile': profile}

        if self.transform:
            sample = self.transform(sample)

        return sample, label

    def get_image_path(self, idx):
        return self.data[idx]

    def set_transform(self, transform=None):
        self.transform = transform

# Custom transform example function with normalization and resizing
class CustomTransform:
    def __init__(self, cube_mean, cube_std, profile_mean, profile_std, resize_shape=(224, 224)):
        self.cube_mean = torch.tensor(cube_mean).view(-1, 1, 1)
        self.cube_std = torch.tensor(cube_std).view(-1, 1, 1)
        self.profile_mean = torch.tensor(profile_mean)
        self.profile_std = torch.tensor(profile_std)
        self.resize_shape = resize_shape
        
        # Define a torchvision Resize transformation for the cube data
        self.resize = transforms.Resize(resize_shape)

    def __call__(self, sample):
        cube = sample['cube']
        profile = sample['profile']
        
        # Resize the cube data
        # The cube is expected to be in the format (C, H, W), where C is the number of channels
        cube = self.resize(cube)
        
        # Normalize cube
        cube = (cube - self.cube_mean) / self.cube_std
        # Normalize profile
        profile = (profile - self.profile_mean) / self.profile_std
        
        return {'cube': cube, 'profile': profile}

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

# Example usage:
if __name__ == "__main__":
    dataset = HyperspectralDataset(root_dir='../../dibasRP/all')

    # Compute statistics for the dataset
    channel_means_cube, channel_stds_cube, channel_means_profile, channel_stds_profile = compute_statistics(dataset)

    # Create custom transform
    custom_transform = CustomTransform(
        cube_mean=channel_means_cube, 
        cube_std=channel_stds_cube, 
        profile_mean=channel_means_profile, 
        profile_std=channel_stds_profile
    )

    # Apply transform to the dataset
    dataset.set_transform(transform=custom_transform)

    # Example DataLoader
    dataloader = DataLoader(dataset, batch_size=4, shuffle=True, num_workers=4)
    for batch in dataloader:
        print(batch)
        break
