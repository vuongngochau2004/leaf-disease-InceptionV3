import os
import torch
import numpy as np
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
from config import Config

class DataManager:
    """Quản lý dữ liệu và data augmentation"""
    def __init__(self, config: Config):
        self.config = config
        self.class_names = None
        self.class_weights = None
    
    def get_train_transforms(self):
        """Tạo transforms cho tập train với data augmentation"""
        return transforms.Compose([
            transforms.Resize(self.config.img_size),
            transforms.RandomHorizontalFlip(),
            transforms.RandomVerticalFlip(0.2),
            transforms.RandomRotation(20),
            transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2),
            transforms.RandomAffine(degrees=0, translate=(0.1, 0.1)),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ])
    
    def get_val_transforms(self):
        """Tạo transforms cho tập validation/test"""
        return transforms.Compose([
            transforms.Resize(self.config.img_size),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ])
    
    def get_display_transforms(self):
        """Tạo transforms cho hiển thị (không normalize)"""
        return transforms.Compose([
            transforms.Resize(self.config.img_size),
            transforms.ToTensor(),
        ])
    
    def create_dataloaders(self):
        """Tạo train và validation dataloaders"""
        train_tfms = self.get_train_transforms()
        val_tfms = self.get_val_transforms()
        
        try:
            train_set = datasets.ImageFolder(self.config.train_dir, transform=train_tfms)
            val_set = datasets.ImageFolder(self.config.val_dir, transform=val_tfms)
        except FileNotFoundError as e:
            print(f"Lỗi: Không tìm thấy thư mục dataset. {e}")
            print(f"Vui lòng kiểm tra đường dẫn: {self.config.train_dir} và {self.config.val_dir}")
            return None, None

        
        self.class_names = train_set.classes
        
        # Tính class weights
        class_counts = np.array([sum(np.array(train_set.targets) == t) 
                                for t in range(len(self.class_names))])
        
        if np.any(class_counts == 0):
            print("Cảnh báo: Có lớp không có mẫu nào trong tập train.")
            class_weights_raw = np.where(class_counts > 0, 1.0 / class_counts, 0.0)
        else:
             class_weights_raw = 1.0 / class_counts

        self.class_weights = class_weights_raw / (class_weights_raw.sum() + 1e-6) * len(self.class_names)
        self.class_weights = torch.FloatTensor(self.class_weights).to(self.config.device)
        
        print(f"Class weights: {self.class_weights.cpu().numpy()}")
        
        train_loader = DataLoader(train_set, batch_size=self.config.batch_size, 
                                 shuffle=True, num_workers=4)
        val_loader = DataLoader(val_set, batch_size=self.config.batch_size, 
                               shuffle=False, num_workers=4)
        
        return train_loader, val_loader
    
    def create_test_dataloader(self):
        """Tạo test dataloader"""
        test_tfms = self.get_val_transforms()
        
        try:
            test_set = datasets.ImageFolder(self.config.test_dir, transform=test_tfms)
        except FileNotFoundError as e:
            print(f"Lỗi: Không tìm thấy thư mục test. {e}")
            print(f"Vui lòng kiểm tra đường dẫn: {self.config.test_dir}")
            return None, None
        
        if self.class_names is None:
            self.class_names = test_set.classes
        
        test_loader = DataLoader(test_set, batch_size=self.config.batch_size,
                                shuffle=False, num_workers=4)
        
        return test_loader, test_set