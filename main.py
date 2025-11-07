import os
import torch
import argparse

from config import Config
from data_loader import DataManager
from model import ModelManager
from trainer import Trainer
from visualizer import Visualizer
from evaluator import Tester
from predictor import Predictor

def main():
    parser = argparse.ArgumentParser(description="Train, test or predict InceptionV3 model")
    parser.add_argument("--mode", choices=["train", "test", "predict"], default="train",
                        help="Chế độ chạy: train, test, hoặc predict")
    parser.add_argument("--model", type=str, default=None,
                        help="Đường dẫn đến file model (.pt) để test hoặc predict")
    parser.add_argument("--image", type=str, help="Đường dẫn đến ảnh để predict")
    args = parser.parse_args()
    
    # Khởi tạo config
    config = Config()
    
    if args.mode == "train":
        print("🚀 Bắt đầu huấn luyện mô hình InceptionV3 (Fine-tuning)")
        
        data_manager = DataManager(config)
        train_loader, val_loader = data_manager.create_dataloaders()
        
        if train_loader is None or data_manager.class_names is None:
             print("❌ Dừng huấn luyện do lỗi tải dữ liệu.")
             return

        print(f"Số lượng lớp: {len(data_manager.class_names)}")
        print(f"Tên các lớp: {data_manager.class_names}")
        
        model_manager = ModelManager(config, len(data_manager.class_names))
        model = model_manager.build_model()
        
        trainer = Trainer(config, model, train_loader, val_loader, 
                         data_manager.class_weights)
        history, preds, labels = trainer.train()
        
        visualizer = Visualizer(config, data_manager.class_names)
        visualizer.save_training_plots(history, preds, labels)
        visualizer.save_classification_report(labels, preds)
        visualizer.save_error_analysis(labels, preds)
        
        # Lưu class_names vào checkpoint
        checkpoint = torch.load(trainer.best_model_path)
        checkpoint['class_names'] = data_manager.class_names
        torch.save(checkpoint, trainer.best_model_path)
        
        print("✅ Quá trình huấn luyện đã hoàn tất!")
    
    elif args.mode == "test":
        print("🧪 Bắt đầu đánh giá mô hình trên tập test")
        
        model_path = args.model if args.model else os.path.join(config.save_dir, "best_model.pt")
        
        if not os.path.exists(model_path):
            print(f"❌ Không tìm thấy model tại: {model_path}")
            return
        
        if not os.path.exists(config.test_dir):
            print(f"❌ Không tìm thấy thư mục test tại: {config.test_dir}")
            return
        
        checkpoint = torch.load(model_path, map_location=config.device)
        class_names = checkpoint.get('class_names', None)
        
        data_manager = DataManager(config)
        
        if class_names is None:
            print("Không tìm thấy 'class_names' trong checkpoint, đang tự động quét thư mục test...")
            test_loader, test_dataset = data_manager.create_test_dataloader()
            class_names = data_manager.class_names
        else:
            print(f"Đã tải {len(class_names)} tên lớp từ checkpoint.")
            data_manager.class_names = class_names
            test_loader, test_dataset = data_manager.create_test_dataloader()
        
        if test_loader is None:
            print("❌ Dừng test do lỗi tải dữ liệu.")
            return

        model_manager = ModelManager(config, len(class_names))
        model_manager.load_model(model_path)
        
        tester = Tester(config, model_manager.model, class_names)
        tester.test(test_loader, test_dataset)
    
    elif args.mode == "predict":
        print("🔮 Bắt đầu dự đoán ảnh đơn lẻ")
        
        if args.image is None:
            print("❌ Vui lòng cung cấp đường dẫn ảnh với tham số --image")
            return
        
        model_path = args.model if args.model else os.path.join(config.save_dir, "best_model.pt")
        
        if not os.path.exists(model_path):
            print(f"❌ Không tìm thấy model tại: {model_path}")
            return
        
        checkpoint = torch.load(model_path, map_location=config.device)
        class_names = checkpoint.get('class_names', None)
        
        if class_names is None:
            print("Không tìm thấy 'class_names' trong checkpoint, đang quét thư mục train/test...")
            scan_dir = config.train_dir if os.path.exists(config.train_dir) else config.test_dir
            if not os.path.exists(scan_dir):
                print(f"❌ Không tìm thấy {config.train_dir} hoặc {config.test_dir} để lấy tên lớp")
                return
            class_names = sorted([d for d in os.listdir(scan_dir) if os.path.isdir(os.path.join(scan_dir, d))])

        if not class_names:
            print("❌ Không thể xác định tên lớp.")
            return
        
        model_manager = ModelManager(config, len(class_names))
        model_manager.load_model(model_path)
        
        predictor = Predictor(config, model_manager.model, class_names)
        predictor.predict(args.image)

if __name__ == "__main__":
    main()