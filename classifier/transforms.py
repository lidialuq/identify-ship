from torchvision import transforms

# training Transforms
train_transforms = transforms.Compose([
    transforms.RandomResizedCrop((224, 224), scale=(0.8, 1.0)),  # Zooming
    transforms.RandomAffine(
        degrees=10, # rotate by up to 10 degrees
        translate=(0.1, 0.1),  # shift
        scale=(0.95, 1.05)     # stretch or compress by up to 5%
    ), 
    transforms.RandomHorizontalFlip(),
    transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2), # unsure about this
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

# no random transforms for val/test sets
val_test_transforms = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])
