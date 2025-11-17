"""
MAIN MODULE - Digital Inspector
Uses trained model: best.pt
Replace your old main.py with this - COMPLETE VERSION
"""

import cv2
import numpy as np
import json
from pathlib import Path
from ultralytics import YOLO
import time


class DigitalInspector:
    """Document inspection using trained YOLOv8 model"""
    
    def __init__(self, model_path='best.pt'):
        """Initialize with trained model"""
        print(f"🔧 Loading model: {model_path}")
        self.model = YOLO(model_path)
        
        self.class_names = {0: 'signature', 1: 'stamp', 2: 'qr_code'}
        self.class_colors = {
            'signature': (0, 255, 0),
            'stamp': (255, 0, 0),
            'qr_code': (0, 0, 255)
        }
        print("✓ Model loaded!")
    
    def process_image(self, image_path, conf_threshold=0.5, save_visualization=True, output_dir='outputs'):
        """Process document image"""
        image = cv2.imread(str(image_path))
        if image is None:
            raise ValueError(f"Cannot read: {image_path}")
        
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        results = self.model.predict(image_path, conf=conf_threshold, verbose=False)
        
        detections = []
        for result in results:
            for box in result.boxes:
                detections.append({
                    'class': self.class_names.get(int(box.cls[0]), 'unknown'),
                    'confidence': float(box.conf[0]),
                    'bbox': [int(x) for x in box.xyxy[0].cpu().numpy()]
                })
        
        summary = {
            'total': len(detections),
            'signatures': sum(1 for d in detections if d['class'] == 'signature'),
            'stamps': sum(1 for d in detections if d['class'] == 'stamp'),
            'qr_codes': sum(1 for d in detections if d['class'] == 'qr_code')
        }
        
        if save_visualization:
            visualized = self.visualize_detections(image_rgb, detections)
            Path(output_dir).mkdir(exist_ok=True)
            cv2.imwrite(f"{output_dir}/detected_{Path(image_path).name}", 
                        cv2.cvtColor(visualized, cv2.COLOR_RGB2BGR))
        
        return {'detections': detections, 'summary': summary}
    
    def visualize_detections(self, image, detections):
        """Draw boxes on image"""
        image_annotated = image.copy()
        for det in detections:
            color = self.class_colors.get(det['class'], (255, 255, 255))
            x1, y1, x2, y2 = det['bbox']
            cv2.rectangle(image_annotated, (x1, y1), (x2, y2), color, 2)
            label = f"{det['class']} {det['confidence']:.2f}"
            cv2.putText(image_annotated, label, (x1, y1-10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
        return image_annotated


if __name__ == "__main__":
    inspector = DigitalInspector()
    print("✓ Ready!")
