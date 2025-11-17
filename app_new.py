"""
WEB APP - Gradio Interface
Uses trained model: best.pt
Replace your old app.py with this
"""

import gradio as gr
import cv2
import numpy as np
from pathlib import Path
from main_new import DigitalInspector

# Initialize inspector with best.pt
inspector = DigitalInspector(model_path='best.pt')

def process_document(image):
    """Process uploaded document"""
    if image is None:
        return None, "Please upload image"
    
    try:
        # Save temp file
        temp_path = Path('temp_upload.jpg')
        cv2.imwrite(str(temp_path), cv2.cvtColor(image, cv2.COLOR_RGB2BGR))
        
        # Process
        result = inspector.process_image(temp_path, save_visualization=False)
        
        # Visualize
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        visualized = inspector.visualize_detections(image_rgb, result['detections'])
        
        # Report
        report = f"""
### 📊 Analysis Results

**Total Detected:** {result['summary']['total']} objects

- ✍️ **Signatures:** {result['summary']['signatures']}
- 🔷 **Stamps/Seals:** {result['summary']['stamps']}
- 📱 **QR Codes:** {result['summary']['qr_codes']}

**Status:** ✅ Analysis Complete
        """
        
        return visualized, report
    
    except Exception as e:
        return None, f"❌ Error: {str(e)}"


# Create Gradio interface
interface = gr.Interface(
    fn=process_document,
    inputs=gr.Image(label="📄 Upload Document", type="numpy"),
    outputs=[
        gr.Image(label="🎯 Detection Result"),
        gr.Markdown(label="📋 Report")
    ],
    title="🔍 Digital Inspector - Armeta AI",
    description="AI-powered construction document inspection",
    examples=[]
)

if __name__ == "__main__":
    print("🚀 Starting web interface...")
    interface.launch(share=False)
