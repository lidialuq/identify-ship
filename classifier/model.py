import torch
import torchvision.models as models
import torch.nn as nn
from dataset import ShipDataset

class CustomEfficientNet:
    def __init__(self, dataset_path: str):
        # Load dataset and get nr of classes to modify the network
        self.dataset = ShipDataset(root_dir=dataset_path, mode='train', transforms=None)
        self.num_classes = len(self.dataset.classes)
        
        self.model = self._initialize_model()
    
    def _initialize_model(self) -> nn.Module:
        # Load pre-trained efficientnet
        efficientnet = models.efficientnet_b0(pretrained=True)

        # Freeze all layers
        for param in efficientnet.parameters():
            param.requires_grad = False
        # Modify last layer to account for number of classes and set parameters to trainable
        efficientnet.classifier[1] = nn.Linear(efficientnet.classifier[1].in_features, self.num_classes)
        #efficientnet.classifier = nn.Linear(efficientnet.classifier.in_features, self.num_classes)
        for param in efficientnet.classifier[1].parameters():
            param.requires_grad = True

        # Set the model to training mode by default
        efficientnet.train()
        
        return efficientnet
    
    def get_model(self) -> nn.Module:
        return self.model
    
    def save_weights(self, path: str):
        torch.save(self.model.state_dict(), path)

    def load_weights(self, path: str):
        self.model.load_state_dict(torch.load(path))
        self.model.eval()  
