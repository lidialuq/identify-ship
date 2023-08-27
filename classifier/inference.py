import torch
from torch.utils.data import DataLoader
import numpy as np 
from tqdm import tqdm

from model import CustomEfficientNet
from dataset import ShipDataset
from transforms import get_transforms

class ModelInference:
    def __init__(self, model_path, dataset_path, device="cuda"):
        self.device = device
        
        # initialize model and dataset
        self.model = CustomEfficientNet(dataset_path=dataset_path)
        self.model.load_weights(model_path)
        self.model = self.model.get_model().to(device).eval()  # set model to evaluation mode
        
        val_transforms = get_transforms('val')
        self.dataset = ShipDataset(root_dir=dataset_path, mode='test', transforms=val_transforms)  
    
    def infer(self):
        results = []
        with torch.no_grad():
            for idx, dic in tqdm(enumerate(self.dataset), total = len(self.dataset)):
                image = dic['image'].to(self.device).unsqueeze(0)   # add batch dimension
                logits = self.model(image)
                # get the predicted class
                _, predicted_class = torch.max(logits, 1)
                predicted_class = predicted_class.item()

                # get the top 5 predictions
                _, top5_classes = torch.topk(logits, 5)
                top5_classes = top5_classes.squeeze().cpu().numpy()
                
                # make dictionary, convert int64 to int32, otherwise json serialization fails
                result = {
                    'image_path': dic['image_path'],
                    'true': {
                        'class_nr': int(dic['class_index']), 
                        'class_name': dic['class_name'],
                    },
                    'predicted': {
                        'class_nr': int(predicted_class),
                        'class_name': self.dataset.classes[predicted_class]
                    },
                    'top_5_predictions': [
                        {'class_nr': int(cls), 'class_name': self.dataset.classes[cls]} for cls in top5_classes
                    ]
                }

                results.append(result)
                
        return results


