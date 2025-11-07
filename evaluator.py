import os
import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix, classification_report
from tqdm import tqdm
from config import Config

class Tester:
    """Quản lý việc đánh giá model trên tập test"""
    def __init__(self, config: Config, model, class_names: list):
        self.config = config
        self.model = model
        self.class_names = class_names
        self.test_results_dir = None
    
    def test(self, test_loader, test_dataset):
        """Đánh giá model trên tập test"""
        self.test_results_dir = os.path.join(os.path.dirname(self.config.save_dir), 
                                             "test_results")
        os.makedirs(self.test_results_dir, exist_ok=True)
        
        self.model.eval()
        
        criterion = nn.CrossEntropyLoss()
        test_loss = 0
        correct = 0
        total = 0
        all_preds = []
        all_labels = []
        all_probs = []
        
        with torch.no_grad():
            for inputs, labels in tqdm(test_loader, desc="Testing"):
                inputs, labels = inputs.to(self.config.device), labels.to(self.config.device)
                outputs = self.model(inputs)
                loss = criterion(outputs, labels)
                
                test_loss += loss.item() * inputs.size(0)
                
                probs = torch.nn.functional.softmax(outputs, dim=1)
                _, preds = probs.max(1)
                
                correct += preds.eq(labels).sum().item()
                total += labels.size(0)
                
                all_preds.extend(preds.cpu().numpy())
                all_labels.extend(labels.cpu().numpy())
                all_probs.extend(probs.cpu().numpy())
        
        test_loss /= len(test_loader.dataset)
        accuracy = 100. * correct / total
        
        print(f"\n📊 Kết quả test:")
        print(f"   - Loss: {test_loss:.4f}")
        print(f"   - Accuracy: {accuracy:.2f}%")
        
        self._save_test_results(all_labels, all_preds, all_probs, 
                               test_loss, accuracy)
        
        return accuracy, test_loss
    
    def _save_test_results(self, labels, preds, probs, loss, accuracy):
        """Lưu các kết quả test"""
        # Confusion matrix
        cm = confusion_matrix(labels, preds)
        cm_norm = cm.astype('float') / (cm.sum(axis=1)[:, np.newaxis] + 1e-6)
        
        plt.figure(figsize=(12, 10))
        
        def custom_format(val, val_norm):
            if val == 0: return ""
            return f"{val_norm:.1%}\n({int(val)})"
        
        annot = np.empty_like(cm, dtype=object)
        for i in range(cm.shape[0]):
            for j in range(cm.shape[1]):
                annot[i, j] = custom_format(cm[i, j], cm_norm[i, j])
        
        ax = sns.heatmap(cm_norm, annot=annot, fmt="", cmap="Blues",
                        xticklabels=self.class_names, yticklabels=self.class_names,
                        vmin=0, vmax=1, annot_kws={"size": 9})
        
        for i in range(len(self.class_names)):
            ax.add_patch(plt.Rectangle((i, i), 1, 1, fill=False, 
                                      edgecolor='black', lw=2))
        
        plt.xlabel("Predicted", fontsize=12)
        plt.ylabel("True", fontsize=12)
        plt.title("Test Confusion Matrix", fontsize=14, fontweight='bold')
        plt.tight_layout()
        plt.savefig(os.path.join(self.test_results_dir, "test_confusion_matrix.png"), dpi=300)
        plt.close()
        
        # Classification report
        report = classification_report(labels, preds, target_names=self.class_names, zero_division=0)
        print(f"\n📋 Classification Report:\n{report}")
        
        with open(os.path.join(self.test_results_dir, "test_classification_report.txt"), "w") as f:
            f.write(report)
        
        print(f"\n✅ Kết quả test đã được lưu tại: {self.test_results_dir}")