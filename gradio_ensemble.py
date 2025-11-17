import gradio as gr
from ultralytics import YOLO
import cv2
import numpy as np
from pathlib import Path
import time
from io import StringIO
import sys
import json
from datetime import datetime


print("=" * 70)
print("LOADING 3-MODEL VOTING ENSEMBLE (ORIGINAL DATA PRIORITY)")
print("=" * 70)

# Твои 3 модели
models = {
    'model_3_original': None,
    'model_2_best_merged': None,
    'model_1_last': None
}

# ОБНОВИ ПУТИ НА СВОИ!
model_paths = {
    'model_3_original': 'org.pt',
    'model_2_best_merged': 'runs/detect/merged_all9/weights/best.pt',
    'model_1_last': 'runs/detect/merged_all9/weights/last.pt'
}

# Загрузить модели
for model_name, path in model_paths.items():
    try:
        models[model_name] = YOLO(path)
        file_size = Path(path).stat().st_size / (1024 * 1024)
        priority = " ⭐ PRIORITY" if model_name == 'model_3_original' else ""
        print(f"✅ {model_name:25s} Loaded: ({file_size:.1f} MB){priority}")
    except Exception as e:
        print(f"⚠️  {model_name:25s} NOT FOUND")

print("\n" + "=" * 70)
print("MODEL STATUS:")
for name, model in models.items():
    status = "✅ LOADED" if model is not None else "❌ NOT FOUND"
    priority = " ⭐ PRIORITY" if name == 'model_3_original' else ""
    print(f"  {name:25s} {status}{priority}")
print("=" * 70 + "\n")

loaded_count = sum(1 for m in models.values() if m is not None)
if loaded_count == 0:
    print("❌ ERROR: No models loaded!")
    exit(1)


def calculate_iou(box1, box2):
    """Calculate IoU between two boxes"""
    x1c, y1c, w1, h1 = box1
    x2c, y2c, w2, h2 = box2
    
    x1_min, y1_min = x1c - w1/2, y1c - h1/2
    x1_max, y1_max = x1c + w1/2, y1c + h1/2
    
    x2_min, y2_min = x2c - w2/2, y2c - h2/2
    x2_max, y2_max = x2c + w2/2, y2c + h2/2
    
    xi_min = max(x1_min, x2_min)
    yi_min = max(y1_min, y2_min)
    xi_max = min(x1_max, x2_max)
    yi_max = min(y1_max, y2_max)
    
    inter_area = max(0, xi_max - xi_min) * max(0, yi_max - yi_min)
    
    box1_area = w1 * h1
    box2_area = w2 * h2
    union_area = box1_area + box2_area - inter_area
    
    return inter_area / union_area if union_area > 0 else 0


def original_priority_detect_detailed(image):
    """
    3-Model Voting with ORIGINAL DATA PRIORITY + Detailed Output
    Returns: (image_with_boxes, detailed_report_text)
    """
    
    # Capture console output for report
    report_lines = []
    
    start_time = time.time()
    
    # === HEADER ===
    report_lines.append("=" * 70)
    report_lines.append("🔍 ENSEMBLE DETECTION REPORT (ORIGINAL DATA PRIORITY)")
    report_lines.append("=" * 70)
    
    detections = {}
    
    # === RUN ORIGINAL MODEL FIRST ⭐ ===
    if models['model_3_original'] is not None:
        try:
            r_orig = models['model_3_original'](
                image, imgsz=640, conf=0.15, iou=0.3, verbose=False
            )
            detections['model_3_original'] = r_orig[0]
            report_lines.append(f"\n⭐ ORIGINAL MODEL (PRIORITY)")
            report_lines.append(f"   Detections: {len(r_orig[0].boxes)}")
        except Exception as e:
            report_lines.append(f"❌ ORIGINAL: {str(e)}")
    
    # === RUN BACKUP MODELS ===
    if models['model_2_best_merged'] is not None:
        try:
            r2 = models['model_2_best_merged'](
                image, imgsz=640, conf=0.2, iou=0.3, verbose=False
            )
            detections['model_2_best_merged'] = r2[0]
            report_lines.append(f"\n📌 BEST_MERGED MODEL (backup)")
            report_lines.append(f"   Detections: {len(r2[0].boxes)}")
        except Exception as e:
            report_lines.append(f"❌ BEST_MERGED: {str(e)}")
    
    if models['model_1_last'] is not None:
        try:
            r1 = models['model_1_last'](
                image, imgsz=640, conf=0.2, iou=0.3, verbose=False
            )
            detections['model_1_last'] = r1[0]
            report_lines.append(f"\n📌 LAST MODEL (backup)")
            report_lines.append(f"   Detections: {len(r1[0].boxes)}")
        except Exception as e:
            report_lines.append(f"❌ LAST: {str(e)}")
    
    # === COLLECT ALL BOXES ===
    all_boxes = []
    for model_name, results in detections.items():
        for box in results.boxes:
            xywh = box.xywh[0].cpu().numpy() if hasattr(box.xywh, 'cpu') else box.xywh[0]
            cls = int(box.cls[0]) if hasattr(box.cls, '__len__') else int(box.cls)
            conf = float(box.conf[0]) if hasattr(box.conf, '__len__') else float(box.conf)
            
            all_boxes.append({
                'model': model_name,
                'xywh': xywh,
                'cls': cls,
                'conf': conf
            })
    
    report_lines.append(f"\n📊 Total raw detections: {len(all_boxes)}")
    
    # === VOTING LOGIC ===
    voted_boxes = []
    used_indices = set()
    
    model_weights = {
        'model_3_original': 2.0,
        'model_2_best_merged': 1.0,
        'model_1_last': 1.0
    }
    
    for i, box1 in enumerate(all_boxes):
        if i in used_indices:
            continue
        
        is_original = box1['model'] == 'model_3_original'
        
        # AUTO-ACCEPT if ORIGINAL found it with high confidence
        if is_original and box1['conf'] > 0.7:
            voted_boxes.append({
                'xywh': box1['xywh'],
                'cls': box1['cls'],
                'conf': box1['conf'],
                'votes': 1,
                'origin': 'ORIGINAL (HIGH CONF)',
                'models': ['model_3_original']
            })
            used_indices.add(i)
            continue
        
        # Find voting partners
        votes = [box1]
        used_indices.add(i)
        
        for j, box2 in enumerate(all_boxes):
            if j <= i or j in used_indices:
                continue
            
            iou = calculate_iou(box1['xywh'], box2['xywh'])
            
            if iou > 0.3 and box1['cls'] == box2['cls']:
                votes.append(box2)
                used_indices.add(j)
        
        # Consensus logic
        should_accept = False
        origin_text = ""
        
        if is_original:
            should_accept = True
            origin_text = f"ORIGINAL"
        elif len(votes) >= 2:
            should_accept = True
            origin_text = f"Consensus ({len(votes)} votes)"
        
        if should_accept:
            avg_xywh = np.mean([v['xywh'] for v in votes], axis=0)
            avg_conf = np.mean([v['conf'] for v in votes])
            
            voted_boxes.append({
                'xywh': avg_xywh,
                'cls': votes[0]['cls'],
                'conf': avg_conf,
                'votes': len(votes),
                'origin': origin_text,
                'models': [v['model'] for v in votes]
            })
    
    # === BUILD REPORT ===
    report_lines.append("\n" + "=" * 70)
    report_lines.append("📋 DETECTION RESULTS")
    report_lines.append("=" * 70)
    
    class_names = {0: 'Signature', 1: 'Stamp', 2: 'QR Code'}
    
    if voted_boxes:
        # Summary
        summary = {
            'total': len(voted_boxes),
            'signatures': sum(1 for b in voted_boxes if b['cls'] == 0),
            'stamps': sum(1 for b in voted_boxes if b['cls'] == 1),
            'qr_codes': sum(1 for b in voted_boxes if b['cls'] == 2)
        }
        
        report_lines.append(f"\n✅ Total detected: {summary['total']} objects")
        report_lines.append(f"   - ✍️ Signatures: {summary['signatures']}")
        report_lines.append(f"   - 🔷 Stamps: {summary['stamps']}")
        report_lines.append(f"   - 📱 QR codes: {summary['qr_codes']}")
        
        # Detailed detections
        report_lines.append("\n" + "-" * 70)
        report_lines.append("DETAILED DETECTION INFO:")
        report_lines.append("-" * 70)
        
        for idx, det in enumerate(voted_boxes, 1):
            x, y, w, h = det['xywh']
            report_lines.append(f"\n {idx}. {class_names.get(det['cls'], 'Unknown').upper()}")
            report_lines.append(f"    ✓ Confidence: {det['conf']:.1%}")
            report_lines.append(f"    ✓ Votes: {det['votes']}")
            report_lines.append(f"    ✓ Source: {det['origin']}")
            report_lines.append(f"    ✓ Position: x={int(x)}, y={int(y)}")
            report_lines.append(f"    ✓ Size: w={int(w)}, h={int(h)}")
    else:
        report_lines.append("\n⚠️ NO OBJECTS DETECTED")
    
    # === DRAW RESULTS ===
    result_img = image.copy()
    
    class_colors = {
        0: (0, 255, 0),      # Green
        1: (255, 0, 0),      # Red
        2: (0, 0, 255)       # Blue
    }
    
    for det in voted_boxes:
        x, y, w, h = det['xywh']
        x1, y1 = int(x - w/2), int(y - h/2)
        x2, y2 = int(x + w/2), int(y + h/2)
        
        cls_id = det['cls']
        color = class_colors.get(cls_id, (255, 255, 0))
        thickness = 3 if 'ORIGINAL' in det['origin'] else 2
        
        cv2.rectangle(result_img, (x1, y1), (x2, y2), color, thickness)
        
        origin_marker = "⭐" if 'ORIGINAL' in det['origin'] else "✓"
        label = f"{class_names.get(cls_id)}{origin_marker} {det['conf']:.0%}"
        
        font = cv2.FONT_HERSHEY_SIMPLEX
        cv2.putText(result_img, label, (x1, y1 - 10), font, 0.6, color, 2)
    
    # === FOOTER ===
    processing_time = time.time() - start_time
    report_lines.append("\n" + "=" * 70)
    report_lines.append(f"⏱️ Processing time: {processing_time:.3f} seconds")
    report_lines.append("=" * 70)
    
    # Join report
    detailed_report = "\n".join(report_lines)
    
    json_data = results_to_json(voted_boxes, processing_time)
    return result_img, detailed_report, json.dumps(json_data, indent=2)


def results_to_json(voted_boxes, processing_time):
    """Convert detection results to JSON format"""
    class_names = {0: 'signature', 1: 'stamp', 2: 'qr_code'}
    
    json_output = {
        "detections": [],
        "summary": {
            "total_detected": len(voted_boxes),
            "signatures": sum(1 for b in voted_boxes if b['cls'] == 0),
            "stamps": sum(1 for b in voted_boxes if b['cls'] == 1),
            "qr_codes": sum(1 for b in voted_boxes if b['cls'] == 2),
            "processing_time_seconds": round(processing_time, 3),
            "model_used": "3-model-ensemble-voting",
            "timestamp": datetime.utcnow().isoformat() + "Z"
        }
    }
    
    for idx, det in enumerate(voted_boxes, 1):
        x, y, w, h = det['xywh']
        json_output["detections"].append({
            "id": idx,
            "class": class_names.get(det['cls'], 'unknown'),
            "confidence": round(float(det['conf']), 3),
            "bbox": {
                "x_center": round(float(x), 1),
                "y_center": round(float(y), 1),
                "width": round(float(w), 1),
                "height": round(float(h), 1)
            },
            "votes": det['votes'],
            "source": det['origin']
        })
    
    return json_output

# ===== GRADIO INTERFACE WITH TWO OUTPUTS =====
# ===== GRADIO INTERFACE WITH TWO OUTPUTS =====
# ===== IMPROVED GRADIO INTERFACE =====
with gr.Blocks(title="🎯 Ensemble Detection Inspector", theme=gr.themes.Soft()) as demo:
    # Header
    gr.Markdown(
        """
        # 🎯 Document Detection Inspector
        **Ensemble AI with Original Data Priority**
        
        Detects: ✍️ Signatures • 🔷 Stamps • 📱 QR Codes
        """
    )
    
    with gr.Row():
        # LEFT COLUMN - Upload & Controls
        with gr.Column(scale=1):
            gr.Markdown("### 📤 Upload Document")
            
            image_input = gr.Image(
                type="numpy",
                label=None,
                height=300,
                sources=["upload", "clipboard"],
                show_label=False
            )
            
            with gr.Row():
                detect_button = gr.Button(
                    "🔍 Detect Objects",
                    variant="primary",
                    size="lg"
                )
                clear_button = gr.Button(
                    "🗑️ Clear",
                    variant="secondary",
                    size="lg"
                )
        
        # RIGHT COLUMN - Report
        with gr.Column(scale=1):
            gr.Markdown("### 📋 Detection Report")
            report_output = gr.Textbox(
                label=None,
                lines=20,
                max_lines=30,
                interactive=False,
                show_label=False,
                placeholder="Detection results will appear here..."
            )
    
    # RESULTS ROW - Full width image output
    gr.Markdown("### 🎯 Detection Results")
    image_output = gr.Image(
        type="numpy",
        label=None,
        height=500,
        show_label=False
    )
    json_output = gr.Textbox(
    label="📊 JSON Results",
    lines=15,
    interactive=False,
    show_label=True,
    placeholder="JSON data will appear here..."
    )

    # Footer info
    gr.Markdown(
        """
        ---
        **ℹ️ How it works:** This system uses 3 AI models with voting consensus. 
        The original model has priority (⭐), backup models confirm detections.
        """
    )
    
    # === EVENT HANDLERS ===
    
    # Detect button click
    detect_button.click(
    fn=original_priority_detect_detailed,
    inputs=image_input,
    outputs=[image_output, report_output, json_output]
    )

    
    # Clear button click
    def clear_all():
        return None, None, ""
    
    clear_button.click(
        fn=clear_all,
        inputs=None,
        outputs=[image_input, image_output, report_output]
    )

if __name__ == "__main__":
    print("\n🚀 Starting Gradio server on http://localhost:7860\n")
    demo.launch(
        share=False,
        server_name="0.0.0.0",
        server_port=7860,
        show_error=True
    )
