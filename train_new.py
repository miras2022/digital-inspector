"""
TRAINING MODULE - Fine-tune or train
Uses trained model: best.pt
Keep or replace your train.py with this
"""

from ultralytics import YOLO
from pathlib import Path

def train_new_model(epochs=50, imgsz=1280, batch=8):
    """Fine-tune your existing trained model"""
    
    print("🚀 Loading your trained model for fine-tuning...")
    model = YOLO('best.pt')
    
    print("⏳ Training...")
    results = model.train(
        data='dataset/data.yaml',
        epochs=epochs,
        imgsz=imgsz,
        batch=batch,
        name='digital_inspector_improved',
        patience=30,
        device=0,
        lr0=0.001
    )
    
    # Save improved model
    model.save('best_improved.pt')
    
    print("✅ Fine-tuning complete!")
    print(f"New model saved: best_improved.pt")
    
    return results


def validate_model():
    """Validate trained model"""
    print("\n📊 Validating model...")
    model = YOLO('best.pt')
    metrics = model.val()
    
    print(f"\n✅ Validation Results:")
    print(f"mAP@0.5:      {metrics.box.map50:.4f}")
    print(f"mAP@0.5-0.95: {metrics.box.map:.4f}")
    print(f"Precision:    {metrics.box.mp:.4f}")
    print(f"Recall:       {metrics.box.mr:.4f}")
    
    return metrics


if __name__ == "__main__":
    # Validate first
    validate_model()
    
    # Or fine-tune
    # train_new_model(epochs=50, imgsz=1280)