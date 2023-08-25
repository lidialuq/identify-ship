import torch
import wandb
import os
import random
import sys
sys.path.append('../')
from settings import PROJECT_ROOT

class Trainer:
    def __init__(self, model, train_loader, val_loader, criterion_train, 
                criterion_val, optimizer, device="cuda"):
        self.model = model.to(device)
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.criterion_train = criterion_train
        self.criterion_val = criterion_val
        self.optimizer = optimizer
        self.device = device
        if wandb.run.name: # If wnadb online
            self.save_model_folder = os.path.join(PROJECT_ROOT, 'trained_models', 
                    wandb.run.name)
        else:
            self.save_model_folder = os.path.join(PROJECT_ROOT, 'trained_models', 
                    str(random.randint(0, 1000000)))

        wandb.watch(self.model)

    # def calculate_accuracy(self, outputs, targets, topk=(1,)):
    #     """Computes the accuracy over the k top predictions for the specified values of k"""
    #     with torch.no_grad():
    #         maxk = max(topk)
    #         batch_size = targets.size(0)

    #         _, pred = outputs.topk(maxk, 1, True, True)
    #         pred = pred.t() # transpose
    #         correct = pred.eq(targets.view(1, -1).expand_as(pred))
    #         res = []
    #         for k in topk:
    #             correct_k = correct[:k].view(-1).float().sum(0, keepdim=True)
    #             res.append(correct_k.mul_(100.0 / batch_size))  
    #     return res

    def train_one_epoch(self):
        self.model.train()
        total_loss = 0
        correct_predictions = 0
        for inputs, targets in self.train_loader:
            inputs, targets = inputs.to(self.device), targets.to(self.device)

            self.optimizer.zero_grad()

            outputs = self.model(inputs)
            loss = self.criterion_train(outputs, targets)
            loss.backward()
            self.optimizer.step()

            total_loss += loss.item()
            _, predicted = outputs.max(1)
            correct_predictions += predicted.eq(targets).sum().item()

        avg_loss = total_loss / len(self.train_loader)
        accuracy = 100 * correct_predictions / len(self.train_loader.dataset)
        return avg_loss, accuracy

    def validate(self):
        self.model.eval()
        total_loss = 0
        correct_predictions = 0
        with torch.no_grad():
            for inputs, targets in self.val_loader:
                inputs, targets = inputs.to(self.device), targets.to(self.device)

                outputs = self.model(inputs)
                loss = self.criterion_val(outputs, targets)
                total_loss += loss.item()

                _, predicted = outputs.max(1)
                correct_predictions += predicted.eq(targets).sum().item()

        avg_loss = total_loss / len(self.val_loader)
        accuracy = 100 * correct_predictions / len(self.val_loader.dataset)
        return avg_loss, accuracy

    def train(self, epochs):
        for epoch in range(epochs):
            train_loss, train_accuracy = self.train_one_epoch()
            val_loss, val_accuracy = self.validate()

            print(f"Epoch: {epoch+1}/{epochs}, Train Loss: {train_loss:.4f}, Train Acc: {train_accuracy:.2f}%, Val Loss: {val_loss:.4f}, Val Acc: {val_accuracy:.2f}%")
            wandb.log({"train": {"loss": train_loss, "accuracy": train_accuracy}, "val": {"loss": val_loss, "accuracy": val_accuracy}})
            # Save the model every 10 epochs
            if (epoch+1) % 10 == 0:
                if not os.path.exists(self.save_model_folder):
                    os.makedirs(self.save_model_folder)
                self.save_model(os.path.join(self.save_model_folder,
                        f"model_epoch_{epoch+1}.pth"))
        self.save_model(os.path.join(self.save_model_folder, "model_final.pth"))
        wandb.finish()

    def save_model(self, path):
        torch.save(self.model.state_dict(), path)