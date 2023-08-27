import torch
from torch import nn
from torch.utils.data import DataLoader
import wandb

from transforms import get_transforms
from dataset import ShipDataset
from trainer import Trainer
from model import CustomEfficientNet

import os
import sys
sys.path.append('../')
from settings import PROJECT_ROOT



# 1. Define dataset path and hyperparameters
root_dir = os.path.join(PROJECT_ROOT, 'data', 'images')
device = torch.device("cuda:0")
config = {
    "learning_rate": 0.001,
    "weight_decay": 5e-4,
    "batch_size": 32,
    "epochs": 1000,
    "data_augmentation": 1.5,
    "dataset": "ship-dataset",
    "architecture": "EfficientNet",
}

# 2. Initialize wandb
wandb.init(
    project="identify-ship",
    config=config,
)

# 3. Initialize transforms, datasets and dataloaders
train_transforms = get_transforms('train', config["data_augmentation"])
val_transforms = get_transforms('val')

train_dataset = ShipDataset(root_dir=root_dir, mode='train', 
        transforms=train_transforms)
val_dataset = ShipDataset(root_dir=root_dir, mode='val', 
        transforms=val_transforms)

train_loader = DataLoader(train_dataset, batch_size=config["batch_size"], 
        shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=config["batch_size"], 
        shuffle=False)

# 4. Setup model, loss, optimizer, and trainer
model = CustomEfficientNet(dataset_path=root_dir)
model = model.get_model().to(device)

weights_tensor = torch.tensor(train_dataset.class_weights, 
        dtype=torch.float).to(device)
criterion_train = nn.CrossEntropyLoss(weight=weights_tensor)
criterion_val = nn.CrossEntropyLoss()

optimizer = torch.optim.Adam(model.parameters(), 
        lr=config["learning_rate"], weight_decay=config["weight_decay"])
scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, 
        config["epochs"])

trainer = Trainer(model, train_loader, val_loader, criterion_train, 
        criterion_val, optimizer, scheduler, device=device)

# 5. Train the model
trainer.train(epochs=config["epochs"])




