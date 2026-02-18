import modal
import sqlite3
import time
import json
from pathlib import Path
from datetime import datetime

# 1. SETUP STORAGE
vol = modal.Volume.from_name("face-model-storage", create_if_missing=True)
WEIGHTS_PATH = Path("/data/custom_model.pth")
DB_PATH = Path("/data/history.db")

# 2. SETUP ENVIRONMENT
image = (
    modal.Image.debian_slim(python_version="3.10")
    .pip_install(
        "torch", "timm", "facenet-pytorch", "numpy", "Pillow", 
        "fastapi", "python-multipart", "faiss-cpu"
    )
)

app = modal.App("face-pro-app", image=image)

# ==========================================
# PART 1: THE BRAIN (GPU + FAISS)
# ==========================================
@app.cls(gpu="T4", scaledown_window=300, volumes={"/data": vol})
class FaceEngine:
    @modal.enter()
    def setup(self):
        import torch
        import timm
        import faiss
        import numpy as np
        from facenet_pytorch import MTCNN

        print("🔧 Initializing AI & Vector DB...")
        self.device = 'cuda'
        self.detector = MTCNN(keep_all=False, device=self.device)
        
        # Load Model
        self.model = timm.create_model('vit_base_patch14_dinov2.lvd142m', pretrained=True, num_classes=0)
        
        vol.reload()
        if WEIGHTS_PATH.exists():
            try:
                state_dict = torch.load(WEIGHTS_PATH, map_location=self.device)
                self.model.load_state_dict(state_dict, strict=False)
                print("✅ Custom Model Loaded")
            except:
                print("⚠️ Load failed, using default")
        
        self.model.to(self.device).eval()
        
        # Transform setup
        data_config = timm.data.resolve_data_config(self.model.pretrained_cfg)
        self.transform = timm.data.create_transform(**data_config, is_training=False)

        # Initialize FAISS (768 dim)
        self.index = faiss.IndexFlatIP(768) 
        print("✅ System Ready.")

    @modal.method()
    def get_embedding(self, image_bytes):
        from PIL import Image
        import io
        import torch
        import faiss
        
        try:
            img = Image.open(io.BytesIO(image_bytes)).convert('RGB')
            img_cropped = self.detector(img)
            if img_cropped is None: return None
            
            # Preprocess
            img_cropped = (img_cropped.permute(1, 2, 0).numpy() + 1) / 2
            img_pil = Image.fromarray((img_cropped * 255).astype('uint8'))
            tensor = self.transform(img_pil).unsqueeze(0).to(self.device)
            
            with torch.no_grad():
                emb = self.model(tensor).cpu().numpy()
            
            faiss.normalize_L2(emb)
            return emb.flatten()
        except Exception as e:
            print(f"Error: {e}")
            return None

# ==========================================
# PART 2: THE WEBSITE & API (CPU)
# ==========================================
@app.function(image=image, volumes={"/data": vol}, min_containers=1)
@modal.asgi_app()
def fastapi_app():
    from fastapi import FastAPI, UploadFile, File
    from fastapi.responses import HTMLResponse
    from pydantic import BaseModel
    
    # Initialize DB
    conn = sqlite3.connect(str(DB_PATH), check_same_thread=False)
    cursor = conn.cursor()
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS history (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            timestamp TEXT,
            filename TEXT,
            prediction TEXT,
            confidence REAL,
            user_feedback TEXT DEFAULT 'pending' 
        )
    ''')
    conn.commit()

    web_app = FastAPI(title="Face AI Pro")

    # --- HTML DASHBOARD CODE ---
    html_content = """
    <!DOCTYPE html>
    <html lang="en">
    <head>
        <meta charset="UTF-8">
        <meta name="viewport" content="width=device-width, initial-scale=1.0">
        <title>Face AI Command Center</title>
        <style>
            :root { --bg: #0f172a; --card: #1e293b; --text: #e2e8f0; --accent: #3b82f6; --green: #22c55e; --red: #ef4444; }
            body { background: var(--bg); color: var(--text); font-family: sans-serif; margin: 0; padding: 20px; }
            .container { max-width: 1000px; margin: 0 auto; }
            .grid { display: grid; grid-template-columns: 1fr 1fr; gap: 20px; margin-top: 20px; }
            .card { background: var(--card); padding: 20px; border-radius: 12px; }
            h1 { color: var(--accent); text-align: center;}
            
            /* Stats */
            .stats-row { display: flex; gap: 20px; justify-content: space-around; margin-bottom: 20px; }
            .stat-box { text-align: center; padding: 15px; background: #0f172a; border-radius: 8px; width: 100%; }
            .stat-number { font-size: 2rem; font-weight: bold; color: var(--accent); }
            
            /* Table */
            table { width: 100%; border-collapse: collapse; margin-top: 10px; }
            th, td { text-align: left; padding: 10px; border-bottom: 1px solid #334155; font-size: 0.9rem; }
            .status-correct { color: var(--green); }
            .status-wrong { color: var(--red); }
            
            /* Buttons */
            button { cursor: pointer; border: none; padding: 8px 12px; border-radius: 6px; font-weight: bold; }
            .btn-upload { background: var(--accent); color: white; width: 100%; padding: 15px; font-size: 1rem; margin-top: 10px;}
            .btn-check { background: var(--green); color: black; }
            .btn-cross { background: var(--red); color: white; }
            
            /* Upload */
            .file-label { display: block; text-align: center; border: 2px dashed #475569; padding: 30px; border-radius: 8px; cursor: pointer; }
            .file-label:hover { border-color: var(--accent); }
        </style>
    </head>
    <body>
        <div class="container">
            <h1>🧠 Face AI Pro</h1>

            <div class="card stats-row">
                <div class="stat-box">
                    <div class="stat-number" id="accuracy">0%</div>
                    <div>Accuracy</div>
                </div>
                <div class="stat-box">
                    <div class="stat-number" id="total-scans">0</div>
                    <div>Total Scans</div>
                </div>
            </div>

            <div class="grid">
                <div class="card">
                    <h2>📸 Live Analysis</h2>
                    <label class="file-label">
                        <input type="file" id="fileInput" accept="image/*" style="display:none">
                        <span id="fileName">Select Image</span>
                    </label>
                    <button class="btn-upload" onclick="uploadImage()">Analyze Face</button>
                    <div id="result" style="margin-top: 20px; display: none;">
                        <h3>Result: <span id="pred-name" style="color: var(--accent)">...</span></h3>
                        <p>Confidence: <span id="pred-conf">...</span></p>
                    </div>
                </div>

                <div class="card">
                    <h2>📜 Recent History</h2>
                    <table id="historyTable">
                        <thead><tr><th>Time</th><th>Prediction</th><th>Feedback</th></tr></thead>
                        <tbody></tbody>
                    </table>
                </div>
            </div>
        </div>

        <script>
            // No URL needed! It uses relative paths automatically.
            async function loadStats() {
                const res = await fetch(`/stats`);
                const data = await res.json();
                document.getElementById('accuracy').innerText = data.accuracy + "%";
                document.getElementById('total-scans').innerText = data.total_checked;
            }

            async function loadHistory() {
                const res = await fetch(`/history`);
                const data = await res.json();
                const tbody = document.querySelector("#historyTable tbody");
                tbody.innerHTML = "";
                data.forEach(row => {
                    const tr = document.createElement("tr");
                    let feedbackHtml = row.feedback === 'pending' ? 
                        `<button class="btn-check" onclick="sendFeedback(${row.id}, true)">✔</button> 
                         <button class="btn-cross" onclick="sendFeedback(${row.id}, false)">✘</button>` : 
                        `<span class="status-${row.feedback}">${row.feedback.toUpperCase()}</span>`;

                    tr.innerHTML = `<td>${row.timestamp.split(' ')[1]}</td><td>${row.prediction}</td><td>${feedbackHtml}</td>`;
                    tbody.appendChild(tr);
                });
            }

            async function uploadImage() {
                const fileInput = document.getElementById('fileInput');
                if (!fileInput.files[0]) return alert("Select a file first");
                const formData = new FormData();
                formData.append("file", fileInput.files[0]);
                
                const btn = document.querySelector(".btn-upload");
                btn.innerText = "Processing...";
                
                try {
                    const res = await fetch(`/predict`, { method: "POST", body: formData });
                    const data = await res.json();
                    document.getElementById('result').style.display = 'block';
                    document.getElementById('pred-name').innerText = data.prediction;
                    document.getElementById('pred-conf').innerText = (data.confidence * 100).toFixed(1) + "%";
                    loadHistory();
                    loadStats();
                } catch (e) { alert("Error connecting to API"); }
                btn.innerText = "Analyze Face";
            }

            async function sendFeedback(id, isCorrect) {
                await fetch(`/feedback`, {
                    method: "POST", headers: { "Content-Type": "application/json" },
                    body: JSON.stringify({ log_id: id, correct: isCorrect })
                });
                loadHistory(); loadStats();
            }

            document.getElementById('fileInput').addEventListener('change', function() {
                document.getElementById('fileName').innerText = this.files[0].name;
            });

            loadHistory(); loadStats();
        </script>
    </body>
    </html>
    """

    class FeedbackRequest(BaseModel):
        log_id: int
        correct: bool

    # --- ENDPOINTS ---
    
    @web_app.get("/")
    def home():
        return HTMLResponse(content=html_content)

    @web_app.post("/predict")
    async def predict(file: UploadFile = File(...)):
        image_bytes = await file.read()
        emb = FaceEngine().get_embedding.remote(image_bytes)
        
        if emb is None: return {"error": "No face detected"}

        # Simulate FAISS Search (Mock Logic)
        prediction = "Unknown"
        confidence = 0.85 

        # Log to DB
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        cursor.execute("INSERT INTO history (timestamp, filename, prediction, confidence) VALUES (?, ?, ?, ?)",
                       (timestamp, file.filename, prediction, float(confidence)))
        conn.commit()
        return {"id": cursor.lastrowid, "prediction": prediction, "confidence": confidence}

    @web_app.get("/history")
    def get_history():
        cursor.execute("SELECT * FROM history ORDER BY id DESC LIMIT 10")
        rows = cursor.fetchall()
        return [{"id": r[0], "timestamp": r[1], "filename": r[2], "prediction": r[3], "confidence": r[4], "feedback": r[5]} for r in rows]

    @web_app.post("/feedback")
    def submit_feedback(data: FeedbackRequest):
        status = "correct" if data.correct else "wrong"
        cursor.execute("UPDATE history SET user_feedback = ? WHERE id = ?", (status, data.log_id))
        conn.commit()
        return {"status": "updated"}

    @web_app.get("/stats")
    def get_stats():
        cursor.execute("SELECT user_feedback, COUNT(*) FROM history WHERE user_feedback != 'pending' GROUP BY user_feedback")
        counts = dict(cursor.fetchall())
        total = counts.get("correct", 0) + counts.get("wrong", 0)
        acc = (counts.get("correct", 0) / total * 100) if total > 0 else 0
        return {"total_checked": total, "accuracy": round(acc, 1)}

    return web_app