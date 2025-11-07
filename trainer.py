import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import ReduceLROnPlateau
from tqdm import tqdm
from config import Config

class Trainer:
    """Quản lý quá trình huấn luyện"""
    def __init__(self, config: Config, model, train_loader, val_loader, class_weights=None):
        self.config = config
        self.model = model
        self.train_loader = train_loader
        self.val_loader = val_loader
        
        self.criterion = nn.CrossEntropyLoss(weight=class_weights)
        
        self.optimizer = optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), 
                                   lr=config.learning_rate, 
                                   weight_decay=config.weight_decay)
        
        self.scheduler = ReduceLROnPlateau(self.optimizer, mode='min', 
                                          factor=0.1, patience=5)
        
        self.best_val_loss = float('inf')
        self.patience_counter = 0
        self.best_model_path = os.path.join(config.save_dir, "best_model.pt")
        
        self.history = {
            "train_loss": [], "val_loss": [], 
            "train_acc": [], "val_acc": []
        }
    
    def train_epoch(self, epoch):
        """Huấn luyện một epoch"""
        self.model.train()
        train_loss, correct, total = 0, 0, 0
        
        pbar = tqdm(self.train_loader, desc=f"Epoch {epoch+1}/{self.config.epochs} [Train]")
        
        for inputs, labels in pbar:
            inputs, labels = inputs.to(self.config.device), labels.to(self.config.device)
            
            self.optimizer.zero_grad()
            outputs = self.model(inputs)
            loss = self.criterion(outputs, labels)
            loss.backward()
            
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
            self.optimizer.step()
            
            train_loss += loss.item() * inputs.size(0)
            _, preds = outputs.max(1)
            correct += preds.eq(labels).sum().item()
            total += labels.size(0)
            
            pbar.set_postfix(loss=loss.item(), acc=correct/total*100)
        
        train_loss /= len(self.train_loader.dataset)
        train_acc = correct / total * 100
        
        return train_loss, train_acc
    
    def validate(self, epoch):
        """Đánh giá trên tập validation"""
        self.model.eval()
        val_loss, correct, total = 0, 0, 0
        all_preds, all_labels = [], []
        
        with torch.no_grad():
            for inputs, labels in tqdm(self.val_loader, 
                                      desc=f"Epoch {epoch+1}/{self.config.epochs} [Val]"):
                inputs, labels = inputs.to(self.config.device), labels.to(self.config.device)
                outputs = self.model(inputs)
                loss = self.criterion(outputs, labels)
                
                val_loss += loss.item() * inputs.size(0)
                _, preds = outputs.max(1)
                correct += preds.eq(labels).sum().item()
                total += labels.size(0)
                
                all_preds.extend(preds.cpu().numpy())
                all_labels.extend(labels.cpu().numpy())
        
        val_loss /= len(self.val_loader.dataset)
        val_acc = correct / total * 100
        
        return val_loss, val_acc, all_preds, all_labels
    
    def save_checkpoint(self, epoch, val_loss, val_acc, is_best=False):
        """Lưu checkpoint"""
        checkpoint = {
            'epoch': epoch,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'scheduler_state_dict': self.scheduler.state_dict(),
            'val_loss': val_loss,
            'val_acc': val_acc
        }
        
        if is_best:
            torch.save(checkpoint, self.best_model_path)
            print(f"✅ Model tốt nhất đã được lưu tại {self.best_model_path}")
        
        if (epoch + 1) % 10 == 0:
            checkpoint_path = os.path.join(self.config.save_dir, f"model_epoch_{epoch+1}.pt")
            torch.save(checkpoint, checkpoint_path)
    
    def train(self):
        """Huấn luyện model"""
        for epoch in range(self.config.epochs):
            train_loss, train_acc = self.train_epoch(epoch)
            val_loss, val_acc, all_preds, all_labels = self.validate(epoch)
            
            self.history["train_loss"].append(train_loss)
            self.history["train_acc"].append(train_acc)
            self.history["val_loss"].append(val_loss)
            self.history["val_acc"].append(val_acc)
            
            print(f"Epoch {epoch+1}/{self.config.epochs}: "
                  f"Train Loss {train_loss:.4f}, Train Acc {train_acc:.2f}% | "
                  f"Val Loss {val_loss:.4f}, Val Acc {val_acc:.2f}%")
            
            self.scheduler.step(val_loss)
            
            improved = val_loss < self.best_val_loss

            if improved:
                self.best_val_loss = val_loss
                self.patience_counter = 0
                self.save_checkpoint(epoch, val_loss, val_acc, is_best=True)
            else:
                self.patience_counter += 1

            if self.patience_counter >= self.config.patience:
                print(f"⚠️ Early stopping sau {self.config.patience} epochs không cải thiện")
                break
            
            self.save_checkpoint(epoch, val_loss, val_acc, is_best=False)
        
        print(f"Đang tải model tốt nhất từ: {self.best_model_path}")
        best_checkpoint = torch.load(self.best_model_path)
        self.model.load_state_dict(best_checkpoint['model_state_dict'])
        print(f"✅ Đã tải mô hình tốt nhất từ epoch {best_checkpoint['epoch']+1}")
        
        # Chạy validate() lần cuối trên model tốt nhất để lấy preds/labels
        _ , _, final_preds, final_labels = self.validate(best_checkpoint['epoch'])

        return self.history, final_preds, final_labels