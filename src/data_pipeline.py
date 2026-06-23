import os
import glob
import random
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image
import yaml

class ConcreteMultiModalDataset(Dataset):
    def __init__(self, image_paths, labels, transform=None):
        self.image_paths = image_paths
        self.labels = labels
        self.transform = transform

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        img_path = self.image_paths[idx]
        label = self.labels[idx]
        
        # Load image safely in RGB format
        try:
            with Image.open(img_path) as img:
                img = img.convert("RGB")
                if self.transform:
                    img = self.transform(img)
            return img, torch.tensor(label, dtype=torch.long)
        except Exception as e:
            # If an image file is corrupted, return a blank tensor placeholder
            return torch.zeros(3, 224, 224), torch.tensor(label, dtype=torch.long)

def build_data_registry(base_raw_dir):
    """
    Parses the distinct folder structures of Mendeley, CrackForest, and SDNET2018
    and maps them to a unified binary registry (0: Healthy, 1: Cracked).
    """
    image_paths = []
    labels = []

    # 1. Parsing Mendeley Concrete Crack Images
    mendeley_path = os.path.join(base_raw_dir, "Concrete Crack Images for Classification")
    if os.path.exists(mendeley_path):
        pos_files = glob.glob(os.path.join(mendeley_path, "Positive", "*.jpg"))
        neg_files = glob.glob(os.path.join(mendeley_path, "Negative", "*.jpg"))
        
        image_paths.extend(pos_files + neg_files)
        labels.extend([1] * len(pos_files) + [0] * len(neg_files))

    # 2. Parsing SDNET2018 (Decks, Pavements, Walls)
    sdnet_path = os.path.join(base_raw_dir, "SDNET2018")
    if os.path.exists(sdnet_path):
        for structure in ["Decks", "Pavements", "Walls"]:
            struct_path = os.path.join(sdnet_path, structure)
            if os.path.exists(struct_path):
                cracked = glob.glob(os.path.join(struct_path, "Cracked", "*.jpg"))
                non_cracked = glob.glob(os.path.join(struct_path, "Non-cracked", "*.jpg"))
                
                image_paths.extend(cracked + non_cracked)
                labels.extend([1] * len(cracked) + [0] * len(non_cracked))

    # 3. Parsing CrackForest Dataset (All items are intrinsically cracked)
    cf_path = os.path.join(base_raw_dir, "CrackForest", "image")
    if os.path.exists(cf_path):
        cf_files = glob.glob(os.path.join(cf_path, "*.jpg"))
        image_paths.extend(cf_files)
        labels.extend([1] * len(cf_files))

    return image_paths, labels

def get_data_loaders(config_path="params.yaml"):
    # Load configuration settings
    with open(config_path, "r") as f:
        config = yaml.safe_load(f)
        
    target_size = tuple(config["data"]["target_size"])
    batch_size = config["train"]["batch_size"]
    
    # Advanced Data Augmentations mimicking field structural conditions
    train_transform = transforms.Compose([
        transforms.Resize(target_size),
        transforms.RandomHorizontalFlip(),
        transforms.RandomVerticalFlip(),
        transforms.RandomRotation(15),
        transforms.ColorJitter(brightness=0.2, contrast=0.2),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    val_test_transform = transforms.Compose([
        transforms.Resize(target_size),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    # Fetch registry
    all_paths, all_labels = build_data_registry("data/raw")
    
    if not all_paths:
        raise ValueError("Data pipeline registry empty. Verify that your dataset paths match your tree structure layout.")
        
    # Combine and shuffle
    combined = list(zip(all_paths, all_labels))
    random.seed(42)
    random.shuffle(combined)
    all_paths, all_labels = zip(*combined)
    
    # Train/Val/Test Split Calculations
    total = len(all_paths)
    train_idx = int(total * config["data"]["train_split"])
    val_idx = train_idx + int(total * config["data"]["val_split"])
    
    # Split paths
    train_paths, train_lbls = all_paths[:train_idx], all_labels[:train_idx]
    val_paths, val_lbls = all_paths[train_idx:val_idx], all_labels[train_idx:val_idx]
    test_paths, test_lbls = all_paths[val_idx:], all_labels[val_idx:]
    
    # Construct PyTorch Datasets
    train_dataset = ConcreteMultiModalDataset(train_paths, train_lbls, transform=train_transform)
    val_dataset = ConcreteMultiModalDataset(val_paths, val_lbls, transform=val_test_transform)
    test_dataset = ConcreteMultiModalDataset(test_paths, test_lbls, transform=val_test_transform)
    
    # Construct DataLoaders
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=2, drop_last=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=2)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=2)
    
    return train_loader, val_loader, test_loader