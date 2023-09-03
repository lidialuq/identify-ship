import os
import random
from typing import Any, Callable, List, Tuple, Optional

import torch
from torchvision import transforms
from torch.utils.data import Dataset
from torch.utils.data import Dataset, random_split, Subset
from PIL import Image
import sys
sys.path.append('../')
from settings import PROJECT_ROOT

class ShipDataset(Dataset):
    def __init__(self, root_dir: str, mode :str, transforms: Optional[Callable] = None):
        """
        Args:
            root_dir (string): Directory with all the images.
            mode (string): 'train', 'val', or 'test'
            transform (callable, optional): Transform to be applied on an image.
        """
        self.root_dir = root_dir
        self.transform = transforms
        self.classes = os.listdir(root_dir)  
        assert mode in ['train', 'val', 'test']
        self.mode = mode
        # list of tuples: (image_path, class_index)
        self.train_filepaths = []
        self.val_filepaths = []
        self.test_filepaths = []
        for class_index, class_name in enumerate(self.classes):
            class_path = os.path.join(root_dir, class_name)
            all_images = [os.path.join(class_path, f) for f in os.listdir(class_path) if f.endswith('.jpg')]
            
            # shuffle with seed, so that train/val/test sets are consistent across runs
            random.Random(42).shuffle(all_images)

            # Assign images to test, val, and train sets
            self.test_filepaths += [(img, class_index) for img in all_images[:3]]
            self.val_filepaths += [(all_images[3], class_index)]
            self.train_filepaths += [(img, class_index) for img in all_images[4:]]

        # class weights to fix class-imbalance
        self.class_weights = self.compute_class_weights()

    def compute_class_weights(self):
        class_counts = [0] * len(self.classes)
        for _, label in self.train_filepaths:
            class_counts[label] += 1
        return [sum(class_counts) / count for count in class_counts]

    def __len__(self):
        if self.mode == 'train':
            return len(self.train_filepaths)
        elif self.mode == 'val':
            return len(self.val_filepaths)
        elif self.mode == 'test':
            return len(self.test_filepaths)
    
    def __getitem__(self, idx):
        if self.mode == 'train':
            img_path, class_index = self.train_filepaths[idx]
        elif self.mode == 'val':
            img_path, class_index = self.val_filepaths[idx]
        elif self.mode == 'test':
            img_path, class_index = self.test_filepaths[idx]

        image = Image.open(img_path)

        # some images are grayscale, convert to RGB
        if image.mode != 'RGB':
            image = image.convert('RGB')

        if self.transform:
            image = self.transform(image)

        dic = {
            'image': image,
            'class_index': class_index,
            'class_name': self.classes[class_index],
            'image_path': img_path
        }
        return dic

if __name__ == '__main__':
    data_path = os.path.join(PROJECT_ROOT, 'data', 'images')
    dataset = ShipDataset(data_path, mode='test')
    print(len(dataset))
