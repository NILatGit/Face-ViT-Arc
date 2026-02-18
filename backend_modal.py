import modal
from pathlib import Path

# 1. Define the Persistent Storage (Volume)
vol = modal.Volume.from_name("face-model-storage", create_if_missing=True)
WEIGHTS_DIR = Path("/data")
WEIGHTS_FILE = WEIGHTS_DIR / "custom_model.pth"

# 2. Define the Cloud Environment
# FIX: specific python_version="3.11" to avoid the Pillow/3.13 build error
image = (
    modal.Image.debian_slim(python_version="3.11")
    .pip_install(
        "torch", 
        "timm", 
        "facenet-pytorch", 
        "numpy", 
        "Pillow"
    )
)

app = modal.App("face-vit-backend", image=image)

# 3. The Main Class
# FIX: Renamed 'container_idle_timeout' to 'scaledown_window' (new Modal 1.0 API)
@app.cls(gpu="T4", scaledown_window=300, volumes={"/data": vol})
class FaceModel:
    @modal.enter()
    def load_model(self):
        import timm
        import torch
        from facenet_pytorch import MTCNN
        
        print("🔧 Initializing Face Engine...")
        self.device = 'cuda'
        
        # Face Detector
        self.detector = MTCNN(keep_all=False, device=self.device)
        
        # Vision Transformer
        print("🧠 Loading Architecture...")
        self.model = timm.create_model(
            'vit_base_patch14_dinov2.lvd142m', 
            pretrained=True, 
            num_classes=0,
            dynamic_img_size=True
        )
        
        # CHECK FOR CUSTOM WEIGHTS IN VOLUME
        if WEIGHTS_FILE.exists():
            print(f"📂 Found custom weights at {WEIGHTS_FILE}! Loading...")
            try:
                state_dict = torch.load(WEIGHTS_FILE, map_location=self.device)
                self.model.load_state_dict(state_dict, strict=False)
                print("✅ Custom weights loaded successfully.")
            except Exception as e:
                print(f"⚠️ Failed to load custom weights: {e}")
                print("Using standard pre-trained weights instead.")
        else:
            print("ℹ️ No custom weights found. Using standard pre-trained weights.")

        self.model.to(self.device)
        self.model.eval()
        
        # Setup Data Transforms
        config = timm.data.resolve_data_config(self.model.pretrained_cfg)
        self.transform = timm.data.create_transform(**config, is_training=False)

    @modal.method()
    def get_embedding(self, image_bytes):
        from PIL import Image
        import io
        import torch
        
        # Convert bytes to image
        img = Image.open(io.BytesIO(image_bytes)).convert('RGB')
        
        # Detect Face
        img_cropped = self.detector(img)
        if img_cropped is None:
            return None
            
        # Prepare for ViT
        # MTCNN returns tensor [-1, 1], convert back to PIL for timm transform
        img_cropped = (img_cropped.permute(1, 2, 0).numpy() + 1) / 2
        img_cropped_pil = Image.fromarray((img_cropped * 255).astype('uint8'))
        
        # Transform & Inference
        tensor = self.transform(img_cropped_pil).unsqueeze(0).to(self.device)
        
        with torch.no_grad():
            emb = self.model(tensor)
            
        return emb.cpu().numpy().flatten()

    @modal.method()
    def save_custom_weights(self, weights_bytes):
        """Endpoint to receive new weights and save them to the volume"""
        print("💾 Receiving new model weights...")
        
        # Write file to the Volume path
        with open(WEIGHTS_FILE, "wb") as f:
            f.write(weights_bytes)
            
        # Force commit to ensure persistence
        vol.commit()
        print("✅ Weights saved to Volume. Restarting container...")
        return True