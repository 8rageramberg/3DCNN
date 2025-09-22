
class VoxelDataset(torch.utils.data.Dataset):
	def __init__(self):
		for label in <synset_ids>:
			synset_path = os.path.join(<root_dir>, label)
			for model_id in os.listdir(synset_path):
				binvox_path = os.path.join(synset_path, model_id, "models", "model_normalized.surface.binvox")
				if os.path.exists(binvox_path):
                    self.samples.append(binvox_path)
                    self.labels.append(label)
	
	def __len__(self):
        return len(self.samples)
		
	def __getitem__(self, idx):
		binvox_path = self.samples[idx]
		
		voxel = read_binvox(binvox_path)
        voxel_tensor = torch.tensor(voxel).unsqueeze(0)  # shape: (1, D, H, W)
		
		label = self.labels[idx]
		
		return voxel_tensor, label
		
dataset = VoxelDataset()

from torch.utils.data import DataLoader
from torch.utils.data import random_split

# 80% training, 20% validation
train_size = int(0.8 * len(dataset))
val_size = len(dataset) - train_size
train_dataset, val_dataset = random_split(dataset, [train_size, val_size])

train_loader = DataLoader(train_dataset, batch_size=4, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=4, shuffle=False)


Graph CNN
from torch.utils.data import DataLoader
from torch.utils.data import random_split
from pytorch3d.datasets.utils import collate_batched_meshes
import torch

shapenet_dataset = ShapeNetCore(SHAPENET_PATH,
    synsets=<category_ids>,
    version=2,
    load_textures=False # Set load_textures to False to avoid texture loading errors
)

# 80% training, 20% validation
train_size = int(0.8 * len(shapenet_dataset))
val_size = len(shapenet_dataset) - train_size
train_dataset, val_dataset = random_split(shapenet_dataset, [train_size, val_size])

# Define a custom collate function to exclude textures
def custom_collate_fn(batch):
    # Filter out the 'textures' key from each dictionary in the batch list
    filtered_batch = [{key: value for key, value in sample.items() if key != 'textures'} for sample in batch]
    # Use PyTorch3D's collate_batched_meshes on the filtered batch
    return collate_batched_meshes(filtered_batch)


train_loader = DataLoader(train_dataset, batch_size=4, shuffle=True, collate_fn=custom_collate_fn)
val_loader = DataLoader(val_dataset, batch_size=4, shuffle=False, collate_fn=custom_collate_fn)

import torch.nn as nn
import torch.nn.functional as F
from pytorch3d.ops import GraphConv
import torch_scatter # Need to import torch_scatter

class SimpleGraphCNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = GraphConv(3, 64)
        self.conv2 = GraphConv(64, 128)
        self.conv3 = GraphConv(128, 256)

        self.fc1 = nn.Linear(256, 128)
        self.fc2 = nn.Linear(128, 5)

    def forward(self, meshes):
        # Get vertex coordinates as initial features
        x = meshes.verts_packed() 
        edges = meshes.edges_packed()

        # Apply graph convolution layers
        x = F.relu(self.conv1(x, edges))
        x = F.relu(self.conv2(x, edges))
        x = F.relu(self.conv3(x, edges))

        # Global feature aggregation per mesh
        num_verts_per_mesh = meshes.num_verts_per_mesh()

        # Create a batch index tensor for scattering
        batch_index = torch.cat([torch.full((num_verts,), i, device=x.device) for i, num_verts in enumerate(num_verts_per_mesh)])

        # Sum features per mesh
        sum_features = torch_scatter.scatter_add(x, batch_index, dim=0, dim_size=len(num_verts_per_mesh))

        # Calculate average features per mesh
        avg_features = sum_features / num_verts_per_mesh.view(-1, 1).float()

        # Apply fully connected layers
        x = F.relu(self.fc1(avg_features))
        x = self.fc2(x)

        return x

training:
    for batch in train_loader:
        meshes = batch['mesh']
        # Get synset_ids from the batch
        synset_ids = batch['synset_id']
		
		# compute loss based on model outputs and labels extracted from synset_ids

PC CNN
Make custom collate function:
from torch.utils.data import DataLoader
from torch.utils.data import random_split
from pytorch3d.datasets.utils import collate_batched_meshes
import torch

# 80% training, 20% validation
train_size = int(0.8 * len(shapenet_dataset))
val_size = len(shapenet_dataset) - train_size
train_dataset, val_dataset = random_split(shapenet_dataset, [train_size, val_size])

# Define a custom collate function to exclude textures and handle varying vertex counts
def custom_collate_fn(batch):
    verts_list = []
    synset_ids_list = []
    max_points = 700 # Fixed number of points for PointNet input

    for sample in batch:
        # Extract vertices
        verts = sample['verts']

        # Handle varying vertex counts by padding
        if verts.shape[0] < max_points:
            # Pad with zeros
            padding_size = max_points - verts.shape[0]
            padding = torch.zeros((padding_size, verts.shape[1]), dtype=verts.dtype, device=verts.device)
            padded_verts = torch.cat([verts, padding], dim=0)
            verts_list.append(padded_verts)
        else:
             verts_list.append(verts[:max_points]) # Simple truncation if > max_points

        # Extract synset_id
        synset_ids_list.append(sample['synset_id'])

    # Stack the padded vertices to create a batch of point clouds
    batched_verts = torch.stack(verts_list, dim=0)

    # Return a dictionary with the batched point clouds and synset_ids
    return {'verts': batched_verts, 'synset_id': synset_ids_list}


train_loader = DataLoader(train_dataset, batch_size=4, shuffle=True, collate_fn=custom_collate_fn)
val_loader = DataLoader(val_dataset, batch_size=4, shuffle=False, collate_fn=custom_collate_fn)

import torch.nn as nn

class PointNet(nn.Module):
    def __init__(self, num_classes):
        super().__init__()
        # Shared MLP layers (applied point-wise)
        self.mlp1 = nn.Sequential(
            nn.Conv1d(3, 64, 1),
            nn.BatchNorm1d(64),
            nn.ReLU()
        )

        # Classification layers
        self.fc1 = nn.Sequential(
            nn.Linear(64, 32),
            nn.BatchNorm1d(32),
            nn.ReLU()
        )
        self.fc2 = nn.Linear(32, 5)

    def forward(self, x):
        # x is expected to be of shape (batch_size, num_points, in_channels)
        # PointNet expects input as (batch_size, in_channels, num_points)
        x = x.transpose(2, 1)

        # Apply shared MLPs
        x = self.mlp1(x)

        # Max pooling across points
        x = torch.max(x, 2)[0]

        # Apply classification layers
        x = self.fc1(x)
        x = self.fc2(x)

        return x
        
train:
        for batch in train_loader:
        # Get point clouds (vertices) and synset_ids from the batch
        point_clouds = batch['verts'].to(device)
        synset_ids = batch['synset_id']
