import os
import json
import torch
from inference import ModelInference
import sys
sys.path.append('../')
from settings import PROJECT_ROOT

model_path = os.path.join(PROJECT_ROOT, 'trained_models', 'final_model.pth') 
dataset_path = os.path.join(PROJECT_ROOT, 'data', 'images')
device = torch.device("cuda:0")

inferer = ModelInference(model_path, dataset_path, device)
results = inferer.infer()

# Save the results to a JSON file
save_path = os.path.join(PROJECT_ROOT, 'results', 'inference_results.json')
with open(save_path, 'w') as outfile:
    json.dump(results, outfile, indent=4)

print(f"Inference results saved to {save_path}")
