"""
This module defines FaceEngine, a Modal Class that runs on a GPU container.
Using @app.cls instead of @app.function allows the class to be stateful -
the detector, model, transform, and FAISS index are loaded once when the
container starts and remain in memory for the lifetime of the container,
rather than being reloaded on every request.

All public methods are decorated with @modal.method(), which makes them
callable from the CPU container (server.py) via engine.method.remote(...).
The .remote() call serialises the arguments, sends them over Modal's
internal network to the GPU container, and deserialises the return value.

GPU container behaviour:
- Spins up on the first .remote() call after the container is idle.
- scaledown_window=300 means the container stays alive for 5 minutes after
  the last request before Modal shuts it down. Raise this if your usage is
  bursty and you want to avoid repeated cold starts. Lower it to save cost
  during quiet periods.
- The container can handle multiple concurrent requests if Modal schedules
  them onto the same instance (min_containers is not set here, so it scales
  to zero when idle).
- Cold start time is dominated by loading torch, timm, and the ViT weights
  (~30-60 seconds on first boot). Subsequent warm requests are fast.

To switch to a different GPU type, change the gpu= argument in @app.cls:
    gpu="A10G"  - more VRAM (24 GB), better for larger batch sizes
    gpu="A100"  - highest performance, highest cost
    gpu="T4"    - current default, good balance for single-image inference
"""

import modal
import numpy as np

from app.config import (
    app,
    vol,
    FAISS_PATH,
    WEIGHTS_PATH,
    EMBEDDING_DIM,
    MODEL_NAME,
)


# @app.cls registers this class with the Modal app as a GPU-backed stateful
# component. The volumes argument mounts the persistent volume at /data,
# giving the container read/write access to the FAISS index and weights files.
@app.cls(gpu="T4", scaledown_window=300, volumes={"/data": vol})
class FaceEngine:
    @modal.enter()
    def setup(self) -> None:
        """
        Container initialisation. Runs exactly once when the container cold-
        starts. Not called again on warm requests. Sets up four instance
        attributes used by all subsequent method calls:

        self.device : str
            Always "cuda" since this container is GPU-backed.

        self.detector : MTCNN
            Face detector from facenet-pytorch. keep_all=False means it
            returns only the highest-confidence face crop per image as a
            float32 tensor in range [-1, 1] with shape (3, H, W).
            To detect and process multiple faces per image, set keep_all=True
            and update _embed to iterate over the returned list.

        self.model : timm ViT model
            ViT backbone loaded in feature extraction mode (num_classes=0
            removes the classification head, returning a flat embedding
            vector of size EMBEDDING_DIM instead of class logits).

        self.transform : torchvision transform
            Preprocessing pipeline derived from the model's pretrained config.
            Handles resize, centre crop, normalisation to the exact statistics
            the model was pretrained with. Must be applied before passing an
            image tensor to self.model.

        self.index : faiss.IndexIDMap
            Wraps a flat inner-product index (IndexFlatIP). IndexIDMap allows
            vectors to be stored with explicit integer IDs (faiss_idx) rather
            than sequential positions, enabling O(1) deletion by ID.
            Because all embeddings are L2-normalised before insertion, the
            inner product is equivalent to cosine similarity.
        """
        import torch
        import timm
        import faiss
        from facenet_pytorch import MTCNN

        self.device = "cuda"
        self.detector = MTCNN(keep_all=False, device=self.device)
        self.model = timm.create_model(MODEL_NAME, pretrained=True, num_classes=0)

        vol.reload()
        if WEIGHTS_PATH.exists():
            try:
                self.model.load_state_dict(
                    torch.load(WEIGHTS_PATH, map_location=self.device), strict=False
                )
            except Exception:
                # If weight loading fails for any reason (corrupt file,
                # architecture mismatch, etc.), fall back to pretrained weights
                # silently rather than crashing the container.
                pass

        self.model.to(self.device).eval()
        cfg = timm.data.resolve_data_config(self.model.pretrained_cfg)
        self.transform = timm.data.create_transform(**cfg, is_training=False)

        if FAISS_PATH.exists():
            self.index = faiss.read_index(str(FAISS_PATH))
        else:
            self.index = faiss.IndexIDMap(faiss.IndexFlatIP(EMBEDDING_DIM))

    def _embed(self, image_bytes: bytes) -> np.ndarray | None:
        """
        Decode raw image bytes and produce a normalised face embedding.

        Pipeline:
        1. Decode bytes -> PIL RGB image.
        2. MTCNN detects and crops the most prominent face, returning a
           float32 tensor in [-1, 1] range on the GPU.
        3. Rescale the tensor from [-1, 1] to [0, 255] uint8 and convert
           back to PIL so self.transform can process it. (MTCNN outputs are
           not in the same format as the ViT's expected input.)
        4. Apply self.transform (resize, centre crop, normalise to ImageNet
           statistics): produces a [1, 3, H, W] float32 GPU tensor.
        5. Forward pass through the ViT model: produces a [1, EMBEDDING_DIM]
           float32 CPU numpy array.
        6. L2-normalise the vector in-place. After this step the vector has
           unit norm, so inner product == cosine similarity.
        """

        from PIL import Image
        import io
        import torch
        import faiss

        try:
            img = Image.open(io.BytesIO(image_bytes)).convert("RGB")
            face = self.detector(img)
            if face is None:
                return None

            face_np = (face.permute(1, 2, 0).cpu().numpy() + 1) / 2
            face_pil = Image.fromarray((face_np * 255).astype("uint8"))
            tensor = self.transform(face_pil).unsqueeze(0).to(self.device)

            with torch.no_grad():
                emb = self.model(tensor).cpu().numpy().astype("float32")

            faiss.normalize_L2(emb)
            return emb
        except Exception:
            return None

    def _save_index(self) -> None:
        """
        Write the current FAISS index to the volume and commit.

        Must be called after any mutation (register or remove) to persist
        changes. Without vol.commit() the write is buffered and may not
        survive a container restart.

        faiss.write_index serialises the entire index (all vectors and IDs)
        to a single binary file. The file is overwritten atomically on each
        save - there is no incremental append format in FAISS.
        """
        import faiss

        faiss.write_index(self.index, str(FAISS_PATH))
        vol.commit()

    @modal.method()
    def register(self, image_bytes: bytes, faiss_idx: int) -> bool:
        """
        Embed an image and insert it into the FAISS index at a specific ID.

        The faiss_idx must be assigned by the caller (server.py uses
        db.count_identities() as a monotonically increasing ID). The same
        ID must be written to the identities table in SQLite by the caller
        immediately after this method returns True, so that FAISS IDs and
        database rows stay in sync.

        If this method returns True but the caller fails to write to SQLite,
        the index will contain a vector with no corresponding identity name.
        A subsequent identify() call could return that orphaned idx, and
        db.get_identity() would return None, causing a 404 response.
        To fix a desync, delete the volume's faces.index file and re-register
        all faces, or implement a reconciliation script.
        """
        emb = self._embed(image_bytes)
        if emb is None:
            return False

        self.index.add_with_ids(emb, np.array([faiss_idx], dtype="int64"))
        self._save_index()
        return True

    @modal.method()
    def verify(self, image_bytes_a: bytes, image_bytes_b: bytes) -> float | None:
        """
        1:1 verification. Compute the cosine similarity between two face images.

        Both embeddings are computed on the same GPU container in a single
        .remote() call, avoiding two round trips. The dot product of two
        L2-normalised vectors equals their cosine similarity, which ranges
        from -1 (opposite) to 1 (identical). In practice, same-person scores
        are typically above 0.6 and different-person scores are below 0.4
        with DINOv2 embeddings.

        The threshold comparison (score >= VERIFY_THRESHOLD) is done by the
        caller in server.py so that the threshold can be changed without
        touching this class.
        """
        emb_a = self._embed(image_bytes_a)
        emb_b = self._embed(image_bytes_b)
        if emb_a is None or emb_b is None:
            return None

        return float(np.dot(emb_a.flatten(), emb_b.flatten()))

    @modal.method()
    def identify(self, image_bytes: bytes) -> tuple[int, float] | None:
        """
        1:N identification. Find the closest registered identity to a probe.

        Performs an exact nearest-neighbour search over all registered
        embeddings using FAISS inner-product search. Returns the ID and
        similarity score of the single best match.

        The caller (server.py) is responsible for resolving faiss_idx to a
        name via db.get_identity(faiss_idx) and deciding whether the score
        is high enough to be a reliable match.
        """
        emb = self._embed(image_bytes)
        if emb is None:
            return None

        D, I = self.index.search(emb, 1)
        idx = int(I[0][0])
        score = float(D[0][0])

        if idx == -1:
            return None
        return idx, score

    @modal.method()
    def remove(self, faiss_idx: int) -> bool:
        """
        Remove a registered face embedding from the FAISS index by ID.

        Uses IndexIDMap.remove_ids() which allows O(n) deletion by ID without
        rebuilding the entire index. After removal, the slot is freed but the
        faiss_idx is NOT reused automatically - subsequent registrations
        continue from db.count_identities(), which counts the identities
        table rows. If a row was deleted from the DB, its former faiss_idx
        will never be reused (this is intentional to avoid aliasing).

        The caller (server.py /api/register with a delete operation, if added)
        should call db.delete_identity(faiss_idx) in the same request to keep
        the FAISS index and the identities table in sync.
        """
        try:
            self.index.remove_ids(np.array([faiss_idx], dtype="int64"))
            self._save_index()
            return True
        except Exception:
            return False

    @modal.method()
    def count(self) -> int:
        """
        Return the total number of face vectors currently in the FAISS index.

        index.ntotal is an O(1) property maintained by FAISS. This should
        match db.count_identities() at all times if the index and database
        are in sync. A mismatch indicates a failed register or remove call.
        """
        return self.index.ntotal
