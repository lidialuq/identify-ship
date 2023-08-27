from torchvision import transforms


def get_transforms(mode, factor=1):
    """
    Args:
        mode (string): 'train', 'val', or 'test'
        factor (float): factor to scale the transforms by
    Returns:
        transforms (callable, optional): Transform to be applied on an image.
    """
    assert mode in ['train', 'val', 'test']

    train_transforms = transforms.Compose([
        transforms.RandomResizedCrop((224, 224), scale=(1-0.2*factor, 1)),  # Zooming
        transforms.RandomAffine(
            degrees=10*factor, # rotate by up to 10 degrees
            translate=(0.1*factor, 0.1*factor),  # shift
            scale=(1-0.05*factor, 1+0.05*factor)     # stretch or compress by up to 5%
        ), 
        transforms.RandomHorizontalFlip(),
        transforms.ColorJitter(brightness=0.2*factor, contrast=0.2*factor, saturation=0.2*factor), # unsure about this
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    val_test_transforms = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    if mode == 'train':
        return train_transforms
    elif mode == 'val' or mode == 'test':
        return val_test_transforms