import pytorch_lightning as pl
from torch.utils.data import Dataset, DataLoader
from PIL import Image
import torch
import torchvision.transforms as transforms
import os
import yaml

class PolypDataset(Dataset):
    def __init__(self, image_dir, transform=None):
        self.image_dir = image_dir
        self.transform = transform
        self.image_files = [f for f in os.listdir(image_dir) if f.endswith(('.jpg', '.png', '.jpeg'))]
        
    def __len__(self):
        return len(self.image_files)
    
    def __getitem__(self, idx):
        img_name = self.image_files[idx]
        img_path = os.path.join(self.image_dir, img_name)
        
        # Load image
        image = Image.open(img_path).convert('RGB')
        
        # Get label from filename (assuming filename contains label information)
        # You might need to adjust this based on your actual data structure
        label = 1 if 'positive' in img_name.lower() else 0
        
        if self.transform:
            image = self.transform(image)
            
        return image, label

class PolypDataModule(pl.LightningDataModule):
    def __init__(self, data_dir, batch_size=32, num_workers=4):
        super().__init__()
        self.data_dir = data_dir
        self.batch_size = batch_size
        self.num_workers = num_workers
        
        # Define transforms
        self.transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                              std=[0.229, 0.224, 0.225])
        ])
        
    def setup(self, stage=None):
        # Load data paths from yaml
        with open('data.yaml', 'r') as f:
            data_config = yaml.safe_load(f)
        
        # Create datasets
        if stage == 'fit' or stage is None:
            self.train_dataset = PolypDataset(
                os.path.join(self.data_dir, 'train/images'),
                transform=self.transform
            )
            self.val_dataset = PolypDataset(
                os.path.join(self.data_dir, 'valid/images'),
                transform=self.transform
            )
            
        if stage == 'test' or stage is None:
            self.test_dataset = PolypDataset(
                os.path.join(self.data_dir, 'test/images'),
                transform=self.transform
            )
    
    def train_dataloader(self):
        return DataLoader(
            self.train_dataset,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=self.num_workers
        )
    
    def val_dataloader(self):
        return DataLoader(
            self.val_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers
        )
    
    def test_dataloader(self):
        return DataLoader(
            self.test_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers
        ) 