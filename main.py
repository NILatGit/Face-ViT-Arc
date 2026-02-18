import modal
from pathlib import Path
import shutil

# 1. SETUP STORAGE (Shared Cloud Hard Drive)
# Both the Web Server and GPU will use this to share the model file
vol = modal.Volume.from_name("face-model-storage", create_if_missing=True)
WEIGHTS_PATH = Path("/data/custom_model.pth")

# 2. SETUP ENVIRONMENT
image = (
    modal.Image.debian_slim(python_version="3.11")
    .pip_install(
        "torch", 
        "timm", 
        "facenet-pytorch", 
        "numpy", 
        "Pillow",
        "gradio"
    )
)

app = modal.App("face-recognition-app", image=image)

# ==========================================
# PART 1: THE BRAIN (GPU)
# ==========================================
@app.cls(gpu="T4", scaledown_window=300, volumes={"/data": vol})
class FaceEngine:
    @modal.enter()
    def load_model(self):
        import timm
        import torch
        from facenet_pytorch import MTCNN
        
        print("🔧 Loading Face Engine on GPU...")
        self.device = 'cuda'
        self.detector = MTCNN(keep_all=False, device=self.device)
        
        # 1. Define Architecture
        self.model = timm.create_model(
            'vit_base_patch14_dinov2.lvd142m', 
            pretrained=True, 
            num_classes=0,
            dynamic_img_size=True
        )
        
        # 2. Check Volume for Custom Weights
        # We reload the volume to ensure we see the latest file
        vol.reload()
        if WEIGHTS_PATH.exists():
            print(f"📂 Found custom weights at {WEIGHTS_PATH}!")
            try:
                state_dict = torch.load(WEIGHTS_PATH, map_location=self.device)
                self.model.load_state_dict(state_dict, strict=False)
                print("✅ Custom fine-tuned weights loaded.")
            except Exception as e:
                print(f"⚠️ Error loading weights: {e}")
                print("Using standard pre-trained weights.")
        else:
            print("ℹ️ No custom weights found. Using standard DINOv2.")

        self.model.to(self.device)
        self.model.eval()
        
        # 3. Setup Transform
        config = timm.data.resolve_data_config(self.model.pretrained_cfg)
        self.transform = timm.data.create_transform(**config, is_training=False)

    @modal.method()
    def get_embedding(self, image_input):
        from PIL import Image
        import torch
        
        if image_input is None: return None
        
        # Convert Numpy (Gradio) -> PIL
        img = Image.fromarray(image_input).convert('RGB')
        
        # Detect Face
        img_cropped = self.detector(img)
        if img_cropped is None: return None
            
        # Prepare for ViT
        img_cropped = (img_cropped.permute(1, 2, 0).numpy() + 1) / 2
        img_cropped_pil = Image.fromarray((img_cropped * 255).astype('uint8'))
        
        tensor = self.transform(img_cropped_pil).unsqueeze(0).to(self.device)
        
        with torch.no_grad():
            emb = self.model(tensor)
            
        return emb.cpu().numpy().flatten()

# ==========================================
# PART 2: THE WEBSITE (CPU)
# ==========================================
# NOTICE: We mount volumes={"/data": vol} here too!
@app.function(image=image, allow_concurrent_inputs=True, volumes={"/data": vol})
@modal.web_server(port=8000)
def web_ui():
    import gradio as gr
    import numpy as np
    import shutil
    
    # Logic for comparing faces
    def compare_faces(img1, img2, threshold):
        if img1 is None or img2 is None:
            return "Please upload both images.", 0.0
        
        # Call the GPU Engine
        emb1 = FaceEngine().get_embedding.remote(img1)
        emb2 = FaceEngine().get_embedding.remote(img2)
        
        if emb1 is None or emb2 is None:
            return "❌ Face not detected in one of the images.", 0.0
            
        score = np.dot(emb1, emb2) / (np.linalg.norm(emb1) * np.linalg.norm(emb2))
        result_text = "✅ MATCH" if score > threshold else "⛔ NO MATCH"
        return f"{result_text} (Score: {score:.4f})", float(score)

    # Logic for uploading new weights
    def upload_weights(file_obj):
        if file_obj is None: 
            return "No file uploaded."
        
        # file_obj.name is the temporary path where Gradio saved the upload
        print(f"Saving new weights from {file_obj}...")
        
        # Copy to the shared Volume path
        shutil.copy(file_obj, WEIGHTS_PATH)
        
        # Force save
        vol.commit()
        
        return "✅ Success! The model file is saved. The next time you run a comparison, the GPU will reload with these new weights."

    # Build the Interface
    with gr.Blocks(title="Face ViT System") as demo:
        gr.Markdown("# 🧠 Vision Transformer Face Recognition")
        
        with gr.Tab("👥 Verify Faces"):
            with gr.Row():
                im1 = gr.Image(label="Face 1", type="numpy") # Send as numpy array
                im2 = gr.Image(label="Face 2", type="numpy")
            
            thresh = gr.Slider(0.0, 1.0, value=0.6, label="Threshold")
            btn = gr.Button("Compare", variant="primary")
            
            with gr.Row():
                lbl = gr.Label(label="Result")
                num = gr.Number(label="Similarity Score")
            
            btn.click(compare_faces, inputs=[im1, im2, thresh], outputs=[lbl, num])

        with gr.Tab("⚙️ Model Manager"):
            gr.Markdown("### Upload Fine-Tuned Weights")
            gr.Markdown("Upload your `.pth` file here. It will replace the current model on the GPU.")
            
            file_input = gr.File(label="Upload Weights (.pth)", file_count="single", type="filepath")
            upload_btn = gr.Button("Upload to Cloud Storage")
            upload_msg = gr.Textbox(label="Status")
            
            upload_btn.click(upload_weights, inputs=file_input, outputs=upload_msg)

    return demo