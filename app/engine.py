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


@app.cls(gpu="T4", scaledown_window=300, volumes={"/data": vol})
class FaceEngine:
    @modal.enter()
    def setup(self) -> None:
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
                pass

        self.model.to(self.device).eval()

        cfg = timm.data.resolve_data_config(self.model.pretrained_cfg)
        self.transform = timm.data.create_transform(**cfg, is_training=False)

        if FAISS_PATH.exists():
            self.index = faiss.read_index(str(FAISS_PATH))
        else:
            self.index = faiss.IndexIDMap(faiss.IndexFlatIP(EMBEDDING_DIM))

    def _embed(self, image_bytes: bytes) -> np.ndarray | None:
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
        import faiss

        faiss.write_index(self.index, str(FAISS_PATH))
        vol.commit()

    @modal.method()
    def register(self, image_bytes: bytes, faiss_idx: int) -> bool:
        emb = self._embed(image_bytes)
        if emb is None:
            return False
        self.index.add_with_ids(emb, np.array([faiss_idx], dtype="int64"))
        self._save_index()
        return True

    @modal.method()
    def verify(self, image_bytes_a: bytes, image_bytes_b: bytes) -> float | None:
        emb_a = self._embed(image_bytes_a)
        emb_b = self._embed(image_bytes_b)
        if emb_a is None or emb_b is None:
            return None
        return float(np.dot(emb_a.flatten(), emb_b.flatten()))

    @modal.method()
    def identify(self, image_bytes: bytes) -> tuple[int, float] | None:
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
        try:
            self.index.remove_ids(np.array([faiss_idx], dtype="int64"))
            self._save_index()
            return True
        except Exception:
            return False

    @modal.method()
    def count(self) -> int:
        return self.index.ntotal
