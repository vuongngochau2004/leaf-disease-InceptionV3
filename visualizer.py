import os
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix, classification_report
from config import Config

class Visualizer:
    """Quản lý việc trực quan hóa kết quả (validation)"""
    def __init__(self, config: Config, class_names: list):
        self.config = config
        self.class_names = class_names
    
    def save_training_plots(self, history: dict, preds: list, labels: list):
        """Lưu biểu đồ training curves và confusion matrix (validation)"""
        # Loss & Accuracy curves
        plt.figure(figsize=(12, 5))
        
        plt.subplot(1, 2, 1)
        plt.plot(history["train_loss"], label="Train Loss")
        plt.plot(history["val_loss"], label="Val Loss")
        plt.xlabel("Epochs")
        plt.ylabel("Loss")
        plt.legend()
        plt.title("Loss Curve")
        plt.grid(True)
        
        plt.subplot(1, 2, 2)
        plt.plot(history["train_acc"], label="Train Acc")
        plt.plot(history["val_acc"], label="Val Acc")
        plt.xlabel("Epochs")
        plt.ylabel("Accuracy (%)")
        plt.legend()
        plt.title("Accuracy Curve")
        plt.grid(True)
        
        plt.tight_layout()
        plt.savefig(os.path.join(self.config.save_dir, "training_curves.png"), dpi=300)
        plt.close()
        
        # Confusion matrix (raw counts)
        cm = confusion_matrix(labels, preds)
        plt.figure(figsize=(10, 8))
        sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
                   xticklabels=self.class_names, yticklabels=self.class_names)
        plt.xlabel("Predicted")
        plt.ylabel("True")
        plt.title("Validation Confusion Matrix (Counts)")
        plt.tight_layout()
        plt.savefig(os.path.join(self.config.save_dir, "validation_confusion_matrix_count.png"), dpi=300)
        plt.close()
        
        # Confusion matrix (normalized)
        cm_norm = cm.astype('float') / (cm.sum(axis=1)[:, np.newaxis] + 1e-6)
        plt.figure(figsize=(10, 8))
        sns.heatmap(cm_norm, annot=True, fmt=".2f", cmap="Blues",
                   xticklabels=self.class_names, yticklabels=self.class_names)
        plt.xlabel("Predicted")
        plt.ylabel("True")
        plt.title("Validation Confusion Matrix (Normalized)")
        plt.tight_layout()
        plt.savefig(os.path.join(self.config.save_dir, "validation_confusion_matrix_percent.png"), dpi=300)
        plt.close()
    
    def save_classification_report(self, labels: list, preds: list):
        """Lưu classification report (validation)"""
        report = classification_report(labels, preds, target_names=self.class_names, zero_division=0)
        with open(os.path.join(self.config.save_dir, "validation_classification_report.txt"), "w") as f:
            f.write(report)
        
        print("✅ Đã lưu: training_curves.png, validation_confusion_matrix_*.png, validation_classification_report.txt")
    
    def save_error_analysis(self, labels: list, preds: list):
        """Phân tích và lưu thông tin về lỗi (validation)"""
        misclassified = np.where(np.array(preds) != np.array(labels))[0]
        
        if len(misclassified) == 0:
            print("✅ Không có lỗi phân loại nào trên tập validation!")
            return
        
        with open(os.path.join(self.config.save_dir, "validation_error_analysis.txt"), "w") as f:
            f.write(f"Tổng số mẫu bị phân loại sai (validation): {len(misclassified)}/{len(preds)} "
                   f"({len(misclassified)/len(preds)*100:.2f}%)\n\n")
            
            for i, class_name in enumerate(self.class_names):
                class_idx = np.where(np.array(labels) == i)[0]
                misclass_idx = np.where((np.array(labels) == i) & (np.array(preds) != i))[0]
                
                if len(class_idx) > 0:
                    error_rate = len(misclass_idx) / len(class_idx) * 100
                    f.write(f"Lớp '{class_name}': {len(misclass_idx)}/{len(class_idx)} "
                           f"mẫu bị phân loại sai ({error_rate:.2f}%)\n")
                    
                    if len(misclass_idx) > 0:
                        pred_classes = np.array(preds)[misclass_idx]
                        unique_preds, counts = np.unique(pred_classes, return_counts=True)
                        sorted_indices = np.argsort(-counts)
                        f.write("  Thường bị nhầm với: ")
                        for idx in sorted_indices[:3]:
                            if idx < len(unique_preds):
                                f.write(f"{self.class_names[unique_preds[idx]]}: {counts[idx]} mẫu "
                                       f"({counts[idx]/len(misclass_idx)*100:.1f}%), ")
                        f.write("\n")
        
        print("✅ Đã lưu validation_error_analysis.txt")