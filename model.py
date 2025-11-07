import torch
import torch.nn as nn
from torchvision import models
from config import Config

class ModelManager:
    """Quản lý mô hình InceptionV3"""
    def __init__(self, config: Config, num_classes: int):
        self.config = config
        self.num_classes = num_classes
        self.model = None
    
    def build_model(self):
        """
        Khởi tạo mô hình InceptionV3 và cài đặt fine-tuning.
        """
        self.model = models.inception_v3(weights=None)

        self.model.aux_logits = False

        # Thay classifier cuối
        in_features = self.model.fc.in_features
        self.model.fc = nn.Linear(in_features, self.num_classes)

        # Khởi tạo lớp fc mới
        nn.init.kaiming_normal_(self.model.fc.weight, mode='fan_in', nonlinearity='relu')
        if self.model.fc.bias is not None:
            nn.init.constant_(self.model.fc.bias, 0)

        print("🔓 Cài đặt fine-tuning: Mở khóa các lớp Mixed_6*, Mixed_7*, fc")
        
        for name, p in self.model.named_parameters():
            if ("Mixed_6" in name) or ("Mixed_7" in name) or ("fc" in name):
                p.requires_grad = True
            else:
                p.requires_grad = False

        self.model = self.model.to(self.config.device)

        # In thông tin tham số
        total_params = sum(p.numel() for p in self.model.parameters())
        trainable_params = sum(p.numel() for p in self.model.parameters() if p.requires_grad)
        print(f"Tổng số tham số: {total_params:,}")
        print(f"Số tham số huấn luyện (Fine-tuning): {trainable_params:,} ({trainable_params/total_params*100:.2f}%)")

        return self.model
    
    def load_model(self, model_path: str):
        """Tải model từ checkpoint"""
        print(f"🔍 Đang tải model từ: {model_path}")
        checkpoint = torch.load(model_path, map_location=self.config.device)
        
        if self.model is None:
            self.model = models.inception_v3(weights=None, aux_logits=False, num_classes=self.num_classes)
        
        state = checkpoint['model_state_dict'] if 'model_state_dict' in checkpoint else checkpoint
        state = {k: v for k, v in state.items() if not k.startswith('AuxLogits.')}

        missing, unexpected = self.model.load_state_dict(state, strict=False)
        print(f"Loaded with strict=False. Missing keys: {len(missing)}, Unexpected keys: {len(unexpected)}")
        
        self.model = self.model.to(self.config.device)
        
        return checkpoint