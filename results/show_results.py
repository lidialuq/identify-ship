import os
import json
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image
import sys
sys.path.append('../')
from settings import PROJECT_ROOT
from classifier.dataset import ShipDataset


def load_results(file_path):
    with open(file_path, 'r') as f:
        return json.load(f)

def average_accuracies(results):
    top1_correct = 0
    top5_correct = 0
    for entry in results:
        true_class = entry['true']['class_nr']
        
        if true_class == entry['predicted']['class_nr']:
            top1_correct += 1
        
        if true_class in [pred['class_nr'] for pred in entry['top_5_predictions']]:
            top5_correct += 1
    
    top1_accuracy = top1_correct / len(results)
    top5_accuracy = top5_correct / len(results)
    
    return top1_accuracy, top5_accuracy

def per_class_acuracy(results, dataset_path):
    top1_accuracy = {}
    top5_accuracy = {}
    class_counts = {}
    
    dataset = ShipDataset(root_dir=dataset_path, mode='train')
    
    for entry in results:
        true_class = entry['true']['class_name']
        
        class_counts[true_class] = class_counts.get(true_class, 0) + 1
        
        if true_class in [pred['class_name'] for pred in entry['top_5_predictions']]:
            top5_accuracy[true_class] = top5_accuracy.get(true_class, 0) + 1

        if true_class == entry['predicted']['class_name']:
            top1_accuracy[true_class] = top1_accuracy.get(true_class, 0) + 1
            
    top1_accuracy = {cls: top1_accuracy.get(cls, 0) / class_counts.get(cls, 1) for cls in dataset.classes}
    top5_accuracy = {cls: top5_accuracy.get(cls, 0) / class_counts.get(cls, 1) for cls in dataset.classes}
    
    top1_accuracy = sorted(top1_accuracy.items(), key=lambda item: item[1])
    top5_accuracy = sorted(top5_accuracy.items(), key=lambda item: item[1])
    
    return top1_accuracy, top5_accuracy

def plot_accuracy_vs_training_images(per_class_acuracy, dataset_path):
    
    dataset = ShipDataset(root_dir=dataset_path, mode='train')
    
    # get nr of training images per class
    #class_counts = [0] * len(dataset.classes)
    class_counts = {}
    for _, label in dataset.train_filepaths:
        # get class name
        class_name = dataset.classes[label]
        class_counts[class_name] = class_counts.get(class_name, 0) + 1
    
    accuracies = []
    image_counts = []
    
    for cls, accuracy in per_class_acuracy:
        accuracies.append(accuracy)
        image_counts.append(class_counts[cls])
    
    plt.figure()
    plt.scatter(image_counts, accuracies)
    plt.xlabel('Number of Training Images')
    plt.ylabel('Top-5 Accuracy')
    plt.title('Top-5 Accuracy vs. Number of Training Images per Class')
    plt.savefig(os.path.join(PROJECT_ROOT, 'results', 'accuracy_vs_nr_images.png'))
    plt.show()


def plot_sample_predictions(results, dataset_path, correct_predictions=True):
    # Load dataset without transforms for raw image access
    test_dataset = ShipDataset(root_dir=dataset_path, mode='test', transforms=None)
    train_dataset = ShipDataset(root_dir=dataset_path, mode='train', transforms=None)

    # Filter results based on the correct_predictions argument
    if correct_predictions:
        filtered_results = [r for r in results if r['predicted']['class_nr'] == r['true']['class_nr']]
        plot_title = 'Correct Predictions'
        save_path = os.path.join(PROJECT_ROOT, 'results', 'examples_correct.png')
    else:
        filtered_results = [r for r in results if r['predicted']['class_nr'] != r['true']['class_nr']]
        plot_title = 'Incorrect Predictions'
        save_path = os.path.join(PROJECT_ROOT, 'results', 'examples_incorrect.png')
    
    # Sample 6 random images from these filtered results
    sampled_indices = np.random.choice(len(filtered_results), size=min(6, len(filtered_results)), replace=False)
    sampled_results = [filtered_results[i] for i in sampled_indices]

    fig, axes = plt.subplots(2, 6, figsize=(15, 5))
    for i, result in enumerate(sampled_results):
        # Display the test image in the first column
        img_path = result['image_path']
        img = Image.open(img_path)
        axes[0, i].imshow(img)
        axes[0, i].axis('off')

        # For the next columns, display three images from the top predicted class
        top_predicted_class_nr = result['predicted']['class_nr']
        top_predicted_class_images = []

        for train_img in train_dataset: 
            if train_img['class_index'] == top_predicted_class_nr:
                top_predicted_class_images.append(train_img['image_path'])
                break
                
        #top_predicted_class_images = test_dataset
        #top_predicted_class_images = [item[0] for item in test_dataset.data if item[1] == top_predicted_class_nr]
        sample_img_path = np.random.choice(top_predicted_class_images)
        sample_img = Image.open(sample_img_path)
        axes[1, i].imshow(sample_img)
        axes[1, i].axis('off')

    # Main title for the plot
    fig.suptitle(plot_title, fontsize=16)

    plt.tight_layout(rect=[0, 0.03, 1, 0.95])  # Adjust layout to accommodate the main title
    plt.savefig(save_path)
    plt.show()

   
def main():
    data_path = os.path.join(PROJECT_ROOT, 'data', 'images')
    results = load_results('inference_results.json')
    
    top1_accuracy, top5_accuracy = average_accuracies(results)
    print(f"Top-1 Accuracy: {top1_accuracy*100:.2f}%")
    print(f"Top-5 Accuracy: {top5_accuracy*100:.2f}%")
    
    top1acc, top5acc = per_class_acuracy(results, data_path)
    print("\n10 Best Performing Classes:")
    for cls, acc in top1acc[-10:]:
        print(f"{cls}: {acc*100:.2f}%")
    
    print("\n10 Worst Performing Classes:")
    for cls, acc in top1acc[:10]:
        print(f"{cls}: {acc*100:.2f}%")
    
    plot_accuracy_vs_training_images(top1acc, data_path)
    # Plot correct predictions
    plot_sample_predictions(results, data_path, correct_predictions=True)

    # Plot incorrect predictions
    plot_sample_predictions(results, data_path, correct_predictions=False)

    

if __name__ == "__main__":
    main()