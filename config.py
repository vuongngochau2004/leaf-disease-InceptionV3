import os
import torch

class Config:
    """Cấu hình cho quá trình huấn luyện"""
    def __init__(self):
        self.train_dir = "Dataset/train"
        self.val_dir = "Dataset/val"
        self.test_dir = "Dataset/test"
        self.img_size = (299, 299)  # Kích thước yêu cầu của InceptionV3
        self.batch_size = 64
        self.epochs = 80
        self.learning_rate = 1e-4 
        self.weight_decay = 1e-4
        self.save_dir = "results"
        self.patience = 12  # Early stopping patience
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        # Tạo thư mục kết quả
        os.makedirs(self.save_dir, exist_ok=True)
        
        print(f"Using device: {self.device}")