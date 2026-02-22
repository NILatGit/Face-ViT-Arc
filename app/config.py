"""
Central configuration for the entire Face-ViT-Arc application.

Every other module in the app/ package imports from here. Nothing in this
file runs application logic - it only declares infrastructure handles,
filesystem paths, and tunable constants.

Changing a value here propagates automatically to all modules that use it,
so this is the only file you need to edit for most configuration changes.
"""

import modal
from pathlib import Path


# Persistent Volume
# A Modal Volume is a networked filesystem that persists across container
# restarts and is accessible from both the GPU (FaceEngine) and CPU
# (fastapi_app) containers simultaneously.
vol = modal.Volume.from_name("face-pro-storage", create_if_missing=True)


# Filesystem Paths (inside the container)
# The volume is always mounted at /data inside both containers. All
# persistent files live under DATA_DIR. Never use relative paths for these
# since the working directory inside a Modal container is not guaranteed.
DATA_DIR = Path("/data")

# SQLite database file. Stores two tables:
#   - identities: maps faiss_idx (int) -> name (str) + registration timestamp
#   - logs: audit trail of every verify/identify/register call
# Managed exclusively by app/db.py.
DB_PATH = DATA_DIR / "logs.db"

# FAISS index file. Stores the face embedding vectors on disk.
# Written by FaceEngine.register() and FaceEngine.remove() via faiss.write_index().
# Read back by FaceEngine.setup() on every container cold start.
# The index type is IndexIDMap(IndexFlatIP(EMBEDDING_DIM)) - see engine.py.
FAISS_PATH = DATA_DIR / "faces.index"

# Optional fine-tuned ViT weights. If this file exists on the volume when
# FaceEngine starts, it is loaded with strict=False on top of the pretrained
# DINOv2 weights. If the file does not exist, the unmodified pretrained
# weights are used. Delete or rename this file on the volume to revert to
# the default pretrained model.
# Upload custom weights with: modal volume put face-pro-storage <local.pth> custom_model.pth
WEIGHTS_PATH = DATA_DIR / "custom_model.pth"


# Secrets
# A Modal Secret is a named key-value store managed in your Modal account.
# Its values are injected as environment variables into whichever container
# the secret is attached to (see main.py secrets=[api_secret]).
#
# Required keys in the "face-api-secret" secret:
#   API_KEY   - The bearer token your API clients must send in the X-API-Key
#               header. Can be any string. Used by app/auth.py.
#   DEV_MODE  - Optional. Set to "true" to bypass API key checks entirely.
#               Useful during local development with `modal serve main.py`.
#               Omit or set to "false" in production.
#
# Create or update the secret:
#   modal secret create face-api-secret API_KEY=<your-key>
#   modal secret create face-api-secret API_KEY=dev DEV_MODE=true   # dev mode
api_secret = modal.Secret.from_name("face-api-secret")


# Container Image
# Defines the Docker-like image used by both the GPU (FaceEngine) and CPU
# (fastapi_app) containers. Both components share the same image so that
# only one image is built and cached per deployment.
#
# debian_slim is a minimal Debian base image. The Python version is pinned
# to 3.10 to match the union type hint syntax (X | Y) used in engine.py.
# If you bump the Python version, verify all type hints remain valid.
#
# pip_install packages:
#   torch           - PyTorch, required by timm and facenet-pytorch
#   timm            - Model zoo; used to load the ViT backbone
#   facenet-pytorch - Provides MTCNN face detector
#   numpy           - Array operations and FAISS input format
#   Pillow          - Image decoding (PIL.Image.open)
#   fastapi         - ASGI web framework for the API server
#   python-multipart- Required by FastAPI to parse multipart/form-data uploads
#   faiss-cpu       - Vector similarity index (CPU build; GPU build not needed
#                     since FAISS runs on the CPU even in the GPU container)

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


# Modal App
# The App object is the top-level Modal construct. It names the deployment
# and groups all functions and classes that belong to it.
app = modal.App("face-recognition-suite", image=image)


# Inference Constants
# The output dimensionality of the ViT embedding model. DINOv2 ViT-Base
# outputs a 768-dimensional vector. If you switch to a different model
# variant (e.g. ViT-Large = 1024, ViT-Small = 384), update this value.
# Changing this invalidates any existing faces.index on the volume - you
# will need to delete it and re-register all faces.
EMBEDDING_DIM = 768

# Cosine similarity threshold for 1:1 verification. Scores range from -1
# to 1 (higher = more similar). A score above this value is classified as
# a match. 0.6 is a reasonable starting point for DINOv2 embeddings.
# Raise it (e.g. 0.7) to reduce false positives at the cost of more false
# negatives. Lower it (e.g. 0.5) to be more permissive.
VERIFY_THRESHOLD = 0.6

# timm model identifier for the ViT backbone.
# "vit_base_patch14_dinov2.lvd142m" is ViT-Base with 14x14 patches,
# pretrained with the DINOv2 self-supervised method on LVD-142M.
# To use a different model, replace this string with any valid timm model
# name that outputs a flat embedding vector (num_classes=0).
# Browse available models at: https://huggingface.co/timm
MODEL_NAME = "vit_base_patch14_dinov2.lvd142m"

# Maximum allowed image upload size in bytes. Requests with a larger payload
# receive a 413 response before the image reaches the engine.
# 10 MB is generous for a face photo; lower it if you want to reduce
# transfer costs or prevent abuse.
MAX_IMAGE_BYTES = 10 * 1024 * 1024  # 10 MB
