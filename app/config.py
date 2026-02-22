import modal
from pathlib import Path

# Volume
vol = modal.Volume.from_name("face-pro-storage", create_if_missing=True)

# Paths inside the container
DATA_DIR = Path("/data")
DB_PATH = DATA_DIR / "logs.db"
FAISS_PATH = DATA_DIR / "faces.index"
WEIGHTS_PATH = DATA_DIR / "custom_model.pth"

# Secrets
api_secret = modal.Secret.from_name("face-api-secret")

# Container image - registers the entire app package
image = (
    modal.Image.debian_slim(python_version="3.10")
    .pip_install(
        "torch",
        "timm",
        "facenet-pytorch",
        "numpy",
        "Pillow",
        "fastapi",
        "python-multipart",
        "faiss-cpu",
    )
    .add_local_python_source("app")
)

app = modal.App("face-recognition-suite", image=image)

# Inference constants
EMBEDDING_DIM = 768
VERIFY_THRESHOLD = 0.6
MODEL_NAME = "vit_base_patch14_dinov2.lvd142m"
MAX_IMAGE_BYTES = 10 * 1024 * 1024  # 10 MB
