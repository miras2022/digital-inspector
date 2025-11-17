"""
DEMO MODULE - Test your model
Uses trained model: best.pt
Keep or replace your demo.py with this
"""

from main_new import DigitalInspector
from pathlib import Path
import time
import json


def demo_single_image(image_path):
    """Demo on single image"""
    print("\n" + "="*70)
    print(f"ANALYZING: {image_path}")
    print("="*70)
    
    inspector = DigitalInspector(model_path='best.pt')
    
    start_time = time.time()
    result = inspector.process_image(image_path, save_visualization=True)
    processing_time = time.time() - start_time
    
    print(f"\n⏱️ Processing time: {processing_time:.3f} seconds")
    print(f"📊 Total detected: {result['summary']['total']} objects")
    print(f" - ✍️ Signatures: {result['summary']['signatures']}")
    print(f" - 🔷 Stamps: {result['summary']['stamps']}")
    print(f" - 📱 QR codes: {result['summary']['qr_codes']}")
    
    if result['detections']:
        print("\n📋 Detection Details:")
        for idx, det in enumerate(result['detections'], 1):
            print(f"\n {idx}. {det['class'].upper()}")
            print(f" Confidence: {det['confidence']:.2%}")
            print(f" Bbox: {det['bbox']}")
    
    return result


def demo_batch(image_dir='test_images', max_images=5):
    """Demo on multiple images"""
    print("\n" + "="*70)
    print("BATCH PROCESSING DEMO")
    print("="*70)
    
    inspector = DigitalInspector(model_path='best.pt')
    
    image_path = Path(image_dir)
    images = list(image_path.glob('*.png'))[:max_images]
    
    if not images:
        images = list(image_path.glob('*.jpg'))[:max_images]
    
    if not images:
        print(f"No images found in {image_dir}")
        return []
    
    total_time = 0
    all_results = []
    
    for idx, img_path in enumerate(images, 1):
        print(f"\n[{idx}/{len(images)}] {img_path.name}")
        
        start_time = time.time()
        result = inspector.process_image(img_path, save_visualization=True)
        processing_time = time.t
