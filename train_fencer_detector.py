from ultralytics import YOLO
import torch
from pathlib import Path

def train_model():
    """Train YOLOv8 model on fencer dataset"""
    # Check if CUDA is available
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    # Initialize model
    model = YOLO('yolov8n.pt')  # Load pretrained YOLOv8n model
    
    # Train the model
    results = model.train(
        data='dataset/data.yaml',
        epochs=100,
        imgsz=640,
        batch=16,
        device=device,
        workers=8,
        patience=50,
        save=True,
        save_period=10,
        cache=True,
        exist_ok=True,
        pretrained=True,
        optimizer='auto',
        verbose=True,
        seed=0,
        deterministic=True,
        single_cls=True,
        rect=False,
        cos_lr=False,
        close_mosaic=10,
        resume=False,
        amp=True,
        fraction=1.0,
        label_smoothing=0.0,
        nbs=64,
        overlap_mask=True,
        mask_ratio=4,
        dropout=0.0,
        val=True,
        plots=True
    )
    
    # Save the best model
    best_model = results.best_model
    best_model.export(format='onnx')
    print("\nTraining complete!")
    print(f"Best model saved as: {best_model.export_dir}/best.pt")
    print(f"ONNX model saved as: {best_model.export_dir}/best.onnx")

if __name__ == "__main__":
    train_model() 