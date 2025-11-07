import os
import time
import torch
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
from torchvision import transforms
from config import Config

class Predictor:
    """Quản lý việc dự đoán ảnh đơn lẻ"""
    def __init__(self, config: Config, model, class_names: list):
        self.config = config
        self.model = model
        self.class_names = class_names
        
        self.transform = transforms.Compose([
            transforms.Resize(config.img_size),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ])
        
        self.display_transform = transforms.Compose([
            transforms.Resize(config.img_size),
            transforms.ToTensor(),
        ])
    
    def predict(self, image_path: str):
        """Dự đoán nhãn cho một ảnh"""
        print(f"\n🔍 Dự đoán ảnh: {image_path}")
        
        if not os.path.exists(image_path):
            print(f"❌ Không tìm thấy file ảnh tại: {image_path}")
            return None, None, None
        
        try:
            image = Image.open(image_path).convert('RGB')
            input_tensor = self.transform(image).unsqueeze(0).to(self.config.device)
            display_tensor = self.display_transform(image)
        except Exception as e:
            print(f"❌ Lỗi khi xử lý ảnh: {str(e)}")
            return None, None, None
        
        self.model.eval()
        start_time = time.time()
        
        with torch.no_grad():
            output = self.model(input_tensor)
            probabilities = torch.nn.functional.softmax(output, dim=1)[0].cpu().numpy()
        
        inference_time = (time.time() - start_time) * 1000
        print(f"\n⏱️ Inference Time: {inference_time:.2f} ms")
        
        top3_indices = probabilities.argsort()[-3:][::-1]
        top3_probs = probabilities[top3_indices]
        top3_classes = [self.class_names[idx] for idx in top3_indices]
        
        predicted_class = top3_classes[0]
        predicted_prob = top3_probs[0]
        
        print(f"\n📊 Kết quả dự đoán:")
        print(f"   - Lớp dự đoán: {predicted_class} (xác suất: {predicted_prob*100:.2f}%)")
        print(f"   - Top-3 lớp: {', '.join(top3_classes)}")
        print(f"   - Top-3 xác suất: {', '.join([f'{p*100:.2f}%' for p in top3_probs])}")
        
        self._visualize_prediction(display_tensor, image_path, 
                                  predicted_class, top3_classes, top3_probs)
        
        return predicted_class, top3_classes, top3_probs
    
    def _visualize_prediction(self, display_tensor, image_path, 
                            predicted_class, top3_classes, top3_probs):
        """Trực quan hóa kết quả dự đoán"""
        prediction_dir = os.path.join(self.config.save_dir, "predictions")
        os.makedirs(prediction_dir, exist_ok=True)
        
        plt.figure(figsize=(10, 6))
        
        plt.subplot(1, 2, 1)
        img_np = display_tensor.numpy().transpose((1, 2, 0))
        img_np = np.clip(img_np, 0, 1)
        plt.imshow(img_np)
        plt.title(f"Input: {os.path.basename(image_path)}")
        plt.axis('off')
        
        plt.subplot(1, 2, 2)
        colors = ['#3498db', '#2ecc71', '#e74c3c']
        bars = plt.bar(top3_classes, top3_probs, color=colors)
        
        for bar, prob in zip(bars, top3_probs):
            plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.02,
                    f'{prob*100:.1f}%', ha='center', va='bottom', fontsize=10)
        
        plt.ylim(0, 1.1)
        plt.title(f"Predicted: {predicted_class}")
        plt.ylabel("Probability")
        plt.xticks(rotation=45, ha='right')
        plt.grid(axis='y', linestyle='--', alpha=0.7)
        
        plt.tight_layout()
        
        base_filename = os.path.splitext(os.path.basename(image_path))[0]
        output_filename = f"pred_{base_filename}_{predicted_class}.png"
        output_path = os.path.join(prediction_dir, output_filename)
        
        plt.savefig(output_path, dpi=300)
        plt.close()
        
        print(f"\n✅ Đã lưu kết quả dự đoán tại: {output_path}")