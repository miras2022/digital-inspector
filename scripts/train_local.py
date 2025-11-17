import os
import shutil
from pathlib import Path
from ultralytics import YOLO
import glob


if __name__ == '__main__':
    print("=" * 60)
    print("LOCAL YOLOV8 TRAINING - MERGED DATASETS")
    print("=" * 60)

    # Change these paths to YOUR paths
    DATASETS_DIR = Path("datasets")
    OUTPUT_DIR = Path("output")

    OUTPUT_DIR.mkdir(exist_ok=True)

    # Merge datasets
    print("\n[1/4] Merging datasets...")
    merged_dir = OUTPUT_DIR / "merged_dataset"
    for subdir in ['images/train', 'labels/train', 'images/val', 'labels/val']:
        (merged_dir / subdir).mkdir(parents=True, exist_ok=True)

    class_mapping = {'signatures': {0: 0}, 'qr': {0: 2}, 'stamps': {0: 1}}

    total_train = 0
    total_val = 0

    for dataset_name in ['signatures_extracted', 'qr_extracted', 'stamps_extracted']:
        dataset_path = DATASETS_DIR / dataset_name
        
        if not dataset_path.exists():
            print(f"  ⚠️  {dataset_name} not found!")
            continue
        
        print(f"  • Merging {dataset_name}...")
        
        # Fix class IDs
        for name, mapping in class_mapping.items():
            if name in dataset_path.name:
                for label_file in dataset_path.rglob('*.txt'):
                    with open(label_file, 'r') as f:
                        lines = f.readlines()
                    
                    new_lines = []
                    for line in lines:
                        line = line.strip()
                        if not line or line.startswith('#'):
                            new_lines.append(line + '\n')
                            continue
                        parts = line.split()
                        if parts and parts[0].isdigit() and int(parts[0]) in mapping:
                            parts[0] = str(mapping[int(parts[0])])
                        new_lines.append(' '.join(parts) + '\n')

                    with open(label_file, 'w') as f:
                        f.writelines(new_lines)
                break
        
        # Copy train
        train_img_dir = dataset_path / 'train' / 'images'
        if train_img_dir.exists():
            for img in train_img_dir.glob('*'):
                if img.is_file() and img.suffix.lower() in ['.jpg', '.png', '.jpeg']:
                    shutil.copy(img, merged_dir / 'images' / 'train')
                    total_train += 1
        
        train_lbl_dir = dataset_path / 'train' / 'labels'
        if train_lbl_dir.exists():
            for lbl in train_lbl_dir.glob('*.txt'):
                shutil.copy(lbl, merged_dir / 'labels' / 'train')
        
        # Copy val
        for val_name in ['valid', 'val']:
            val_img_dir = dataset_path / val_name / 'images'
            if val_img_dir.exists():
                for img in val_img_dir.glob('*'):
                    if img.is_file() and img.suffix.lower() in ['.jpg', '.png', '.jpeg']:
                        dst = merged_dir / 'images' / 'val' / img.name
                        if not dst.exists():
                            shutil.copy(img, dst)
                            total_val += 1
            
            val_lbl_dir = dataset_path / val_name / 'labels'
            if val_lbl_dir.exists():
                for lbl in val_lbl_dir.glob('*.txt'):
                    dst = merged_dir / 'labels' / 'val' / lbl.name
                    if not dst.exists():
                        shutil.copy(lbl, dst)

    print(f"  ✓ Train: {total_train} images, Val: {total_val} images")

    # If no validation data, split
    if total_val == 0:
        print("  • Creating validation set (20%)...")
        train_images = list((merged_dir / 'images' / 'train').glob('*'))
        split_idx = int(len(train_images) * 0.2)
        
        for img in train_images[:split_idx]:
            shutil.move(img, merged_dir / 'images' / 'val')
        
        train_labels = list((merged_dir / 'labels' / 'train').glob('*.txt'))
        for lbl in train_labels[:split_idx]:
            shutil.move(lbl, merged_dir / 'labels' / 'val')
        
        total_train -= split_idx
        total_val = split_idx

    # Create data.yaml
    print("\n[2/4] Creating data.yaml...")
    data_yaml = f"""path: {merged_dir.absolute()}
train: images/train
val: images/val
nc: 3
names: ['signature', 'stamp', 'qr_code']
"""
    with open(merged_dir / 'data.yaml', 'w') as f:
        f.write(data_yaml)

    # Train
    print("\n[3/4] Training YOLOv8...")
    model = YOLO('yolov8m.pt')
    results = model.train(
        data=str(merged_dir / 'data.yaml'),
        epochs=100,
        imgsz=640,
        batch=4,
        patience=30,
        device=0,
        name='merged_all'
    )

    # Results
    print("\n[4/4] Complete!")
    metrics = model.val()
    print(f"\nmAP@0.5:   {metrics.box.map50:.3f}")
    print(f"Precision: {metrics.box.mp:.3f}")
    print(f"Recall:    {metrics.box.mr:.3f}")
    print(f"\n✅ Model saved: {OUTPUT_DIR}/runs/detect/merged_all/weights/best.pt")
