import modal
from pathlib import Path

# 1. SETUP STORAGE
vol = modal.Volume.from_name("face-model-storage", create_if_missing=True)
WEIGHTS_PATH = Path("/data/custom_model.pth")

# 2. SETUP ENVIRONMENT
# We only need the essentials. No heavy UI libraries.
image = (
    modal.Image.debian_slim(python_version="3.10")
    .pip_install(
        "torch",
        "timm",
        "facenet-pytorch",
        "numpy",
        "Pillow",
        "fastapi",          # Standard API framework
        "python-multipart"  # Required for file uploads
    )
)

app = modal.App("face-recognition-api", image=image)

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

        print("🔧 GPU: Initializing...")
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
        vol.reload()
        if WEIGHTS_PATH.exists():
            print(f"📂 Found custom weights! Loading from {WEIGHTS_PATH}")
            try:
                state_dict = torch.load(WEIGHTS_PATH, map_location=self.device)
                self.model.load_state_dict(state_dict, strict=False)
                print("✅ Custom weights loaded.")
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
    def get_embedding(self, image_bytes):
        from PIL import Image
        import io
        import torch

        # Convert bytes directly to image
        img = Image.open(io.BytesIO(image_bytes)).convert('RGB')
        
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
# PART 2: THE API (CPU)
# ==========================================
@app.function(image=image, volumes={"/data": vol}, min_containers=1)
@modal.asgi_app()
def fastapi_app():
    from fastapi import FastAPI, UploadFile, File
    import numpy as np
    import shutil
    
    web_app = FastAPI(title="Face ViT API")

    @web_app.post("/compare")
    async def compare_faces(
        file1: UploadFile = File(...), 
        file2: UploadFile = File(...)
    ):
        # Read bytes
        bytes1 = await file1.read()
        bytes2 = await file2.read()
        
        # Send to GPU
        emb1 = FaceEngine().get_embedding.remote(bytes1)
        emb2 = FaceEngine().get_embedding.remote(bytes2)
        
        if emb1 is None or emb2 is None:
            return {"match": False, "score": 0.0, "error": "No face detected in one of the images"}
            
        # Calc Score
        score = np.dot(emb1, emb2) / (np.linalg.norm(emb1) * np.linalg.norm(emb2))
        
        # Return JSON
        return {
            "match": bool(score > 0.6),
            "score": float(score),
            "verdict": "MATCH" if score > 0.6 else "NO MATCH"
        }

    @web_app.post("/upload-model")
    async def upload_model(file: UploadFile = File(...)):
        with open(WEIGHTS_PATH, "wb") as buffer:
            shutil.copyfileobj(file.file, buffer)
        vol.commit()
        return {"status": "success", "message": "Model weights updated. GPU will reload on next request."}

    return web_app