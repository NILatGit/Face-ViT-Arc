import modal
from pathlib import Path

# 1. SETUP STORAGE (Cloud Hard Drive)
vol = modal.Volume.from_name("face-model-storage", create_if_missing=True)
WEIGHTS_PATH = Path("/data/custom_model.pth")

# 2. SETUP ENVIRONMENT (Libraries)
image = (
    modal.Image.debian_slim(python_version="3.11")
    .pip_install(
        "torch", 
        "timm", 
        "facenet-pytorch", 
        "numpy", 
        "Pillow",
        "gradio"  # <--- We install the website builder here
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
        
        print("🔧 Loading Model on GPU...")
        self.device = 'cuda'
        self.detector = MTCNN(keep_all=False, device=self.device)
        
        # Load Architecture
        self.model = timm.create_model(
            'vit_base_patch14_dinov2.lvd142m', 
            pretrained=True, 
            num_classes=0,
            dynamic_img_size=True
        )
        
        # Check for Custom Weights
        if WEIGHTS_PATH.exists():
            print(f"📂 Found custom weights! Loading from {WEIGHTS_PATH}")
            try:
                state_dict = torch.load(WEIGHTS_PATH, map_location=self.device)
                self.model.load_state_dict(state_dict, strict=False)
                print("✅ Custom weights active.")
            except Exception as e:
                print(f"⚠️ Load failed: {e}. Using default.")
        else:
            print("ℹ️ Using default DINOv2 weights.")

        self.model.to(self.device)
        self.model.eval()
        
        # Transform
        config = timm.data.resolve_data_config(self.model.pretrained_cfg)
        self.transform = timm.data.create_transform(**config, is_training=False)

    @modal.method()
    def get_embedding(self, image_input):
        # Gradio sends images as numpy arrays, we need PIL
        from PIL import Image
        import numpy as np
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

    @modal.method()
    def save_weights(self, file_path):
        """Moves uploaded file to persistent volume"""
        import shutil
        print(f"💾 Saving new weights...")
        # Copy from temporary upload spot to persistent volume
        shutil.copy(file_path, WEIGHTS_PATH)
        vol.commit()
        return "✅ Success! Restarting system to apply changes..."

# ==========================================
# PART 2: THE WEBSITE (CPU)
# ==========================================
@app.function(image=image, allow_concurrent_inputs=True)
@modal.web_server(port=8000)
def web_ui():
    import gradio as gr
    import numpy as np
    
    # Define the logic function that Gradio will call
    def compare_faces(img1, img2, threshold):
        if img1 is None or img2 is None:
            return "Please upload both images.", 0.0
        
        # Call the GPU Engine remotely
        # This is the "magic" - it calls the other class internally
        emb1 = FaceEngine().get_embedding.remote(img1)
        emb2 = FaceEngine().get_embedding.remote(img2)
        
        if emb1 is None or emb2 is None:
            return "❌ Face not detected in one of the images.", 0.0
            
        # Cosine Similarity
        score = np.dot(emb1, emb2) / (np.linalg.norm(emb1) * np.linalg.norm(emb2))
        
        result_text = "✅ MATCH" if score > threshold else "⛔ NO MATCH"
        return f"{result_text} (Score: {score:.4f})", float(score)

    def upload_file(file_obj):
        if file_obj is None: return "No file uploaded."
        # Call GPU Engine to save file
        msg = FaceEngine().save_weights.remote(file_obj.name)
        return msg

    # Build the Interface
    with gr.Blocks(title="Face ViT System") as demo:
        gr.Markdown("# 🧠 Vision Transformer Face Recognition")
        
        with gr.Tab("👥 Verify Faces"):
            with gr.Row():
                im1 = gr.Image(label="Face 1")
                im2 = gr.Image(label="Face 2")
            
            thresh = gr.Slider(0.0, 1.0, value=0.6, label="Threshold")
            btn = gr.Button("Compare", variant="primary")
            
            with gr.Row():
                lbl = gr.Label(label="Result")
                num = gr.Number(label="Similarity Score")
            
            btn.click(compare_faces, inputs=[im1, im2, thresh], outputs=[lbl, num])

        with gr.Tab("⚙️ Model Manager"):
            gr.Markdown("Upload your fine-tuned `.pth` file here.")
            file_input = gr.File(label="Upload Weights (.pth)", file_types=[".pth", ".pt"])
            upload_btn = gr.Button("Upload to Cloud Storage")
            upload_msg = gr.Textbox(label="Status")
            
            upload_btn.click(upload_file, inputs=file_input, outputs=upload_msg)

    return demo