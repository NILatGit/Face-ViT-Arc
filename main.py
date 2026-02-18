import modal
import sqlite3
import json
import numpy as np
import shutil
from pathlib import Path
from datetime import datetime

# ==========================================
# 0. CONFIGURATION
# ==========================================
vol = modal.Volume.from_name("face-pro-storage", create_if_missing=True)
DATA_DIR = Path("/data")
DB_PATH = DATA_DIR / "logs.db"
FAISS_PATH = DATA_DIR / "faces.index"
NAMES_PATH = DATA_DIR / "names.json"
WEIGHTS_PATH = DATA_DIR / "custom_model.pth"

image = (
    modal.Image.debian_slim(python_version="3.10")
    .pip_install(
        "torch", "timm", "facenet-pytorch", "numpy", "Pillow", 
        "fastapi", "python-multipart", "faiss-cpu"
    )
)

app = modal.App("face-recognition-suite", image=image)

# ==========================================
# PART 1: THE AI ENGINE (GPU)
# ==========================================
@app.cls(gpu="T4", scaledown_window=300, volumes={"/data": vol})
class FaceEngine:
    @modal.enter()
    def setup(self):
        import torch
        import timm
        import faiss
        from facenet_pytorch import MTCNN

        print("🔧 Initializing Engine...")
        self.device = 'cuda'
        self.detector = MTCNN(keep_all=False, device=self.device)
        
        # Load Model
        self.model = timm.create_model('vit_base_patch14_dinov2.lvd142m', pretrained=True, num_classes=0)
        
        # Load Custom Weights if they exist
        vol.reload()
        if WEIGHTS_PATH.exists():
            try:
                self.model.load_state_dict(torch.load(WEIGHTS_PATH, map_location=self.device), strict=False)
                print("✅ Custom Model Loaded")
            except:
                print("⚠️ Failed to load custom weights, using default.")
        
        self.model.to(self.device).eval()
        
        # Transform
        cfg = timm.data.resolve_data_config(self.model.pretrained_cfg)
        self.transform = timm.data.create_transform(**cfg, is_training=False)

        # Load FAISS Index (The Database of Faces)
        if FAISS_PATH.exists() and NAMES_PATH.exists():
            print("📂 Loading existing Face Database...")
            self.index = faiss.read_index(str(FAISS_PATH))
            with open(NAMES_PATH, 'r') as f:
                self.names = json.load(f)
        else:
            print("mw New Face Database created.")
            self.index = faiss.IndexFlatIP(768) # 768 dim for Base ViT
            self.names = []

    def _process_image(self, image_bytes):
        from PIL import Image
        import io
        import torch
        try:
            img = Image.open(io.BytesIO(image_bytes)).convert('RGB')
            face = self.detector(img)
            if face is None: return None
            
            # Preprocess for ViT
            face = (face.permute(1, 2, 0).numpy() + 1) / 2
            face_pil = Image.fromarray((face * 255).astype('uint8'))
            tensor = self.transform(face_pil).unsqueeze(0).to(self.device)
            
            with torch.no_grad():
                emb = self.model(tensor).cpu().numpy()
            
            # Normalize for Cosine/Inner Product
            import faiss
            faiss.normalize_L2(emb)
            return emb
        except Exception as e:
            print(f"Processing Error: {e}")
            return None

    @modal.method()
    def get_embedding_remote(self, image_bytes):
        emb = self._process_image(image_bytes)
        return emb.flatten() if emb is not None else None

    @modal.method()
    def register_new_face(self, image_bytes, name):
        import faiss
        emb = self._process_image(image_bytes)
        if emb is None: return False, "No face detected"
        
        # Add to FAISS
        self.index.add(emb)
        self.names.append(name)
        
        # Save to Disk immediately
        faiss.write_index(self.index, str(FAISS_PATH))
        with open(NAMES_PATH, 'w') as f:
            json.dump(self.names, f)
        vol.commit()
        
        return True, f"Registered {name} successfully."

    @modal.method()
    def search_face(self, image_bytes):
        emb = self._process_image(image_bytes)
        if emb is None: return None, 0.0
        
        # Search Top 1
        D, I = self.index.search(emb, 1)
        idx = I[0][0]
        score = D[0][0]
        
        if idx == -1: return "Unknown", 0.0
        
        name = self.names[idx]
        return name, float(score)

# ==========================================
# PART 2: THE WEB SERVER (CPU)
# ==========================================
@app.function(image=image, volumes={"/data": vol}, min_containers=1)
@modal.asgi_app()
def fastapi_app():
    from fastapi import FastAPI, UploadFile, File, Form
    from fastapi.responses import HTMLResponse
    from pydantic import BaseModel
    
    # Init Database
    conn = sqlite3.connect(str(DB_PATH), check_same_thread=False)
    c = conn.cursor()
    c.execute('''CREATE TABLE IF NOT EXISTS logs 
                 (id INTEGER PRIMARY KEY, type TEXT, result TEXT, confidence REAL, feedback TEXT, timestamp TEXT)''')
    conn.commit()

    web = FastAPI()

    # --- HTML UI ---
    HTML_CONTENT = """
    <!DOCTYPE html>
    <html>
    <head>
        <title>Face AI Suite</title>
        <style>
            :root { --bg: #0f172a; --card: #1e293b; --text: #f1f5f9; --blue: #3b82f6; --green: #22c55e; }
            body { background: var(--bg); color: var(--text); font-family: sans-serif; margin: 0; padding: 20px; }
            .container { max-width: 900px; margin: 0 auto; }
            
            /* Tabs */
            .tabs { display: flex; gap: 10px; margin-bottom: 20px; }
            .tab-btn { background: var(--card); color: #94a3b8; padding: 12px 20px; border: none; cursor: pointer; border-radius: 8px; font-weight: bold; flex: 1; }
            .tab-btn.active { background: var(--blue); color: white; }
            .tab-content { display: none; }
            .tab-content.active { display: block; }
            
            /* Cards */
            .card { background: var(--card); padding: 25px; border-radius: 12px; margin-bottom: 20px; border: 1px solid #334155; }
            h2 { margin-top: 0; color: var(--blue); }
            
            /* Forms */
            .file-box { border: 2px dashed #475569; padding: 20px; text-align: center; border-radius: 8px; margin: 10px 0; cursor: pointer; }
            .file-box:hover { border-color: var(--blue); }
            button.action { width: 100%; padding: 15px; background: var(--blue); color: white; border: none; border-radius: 8px; font-size: 1rem; cursor: pointer; margin-top: 10px; }
            input[type="text"] { width: 100%; padding: 12px; border-radius: 6px; border: 1px solid #475569; background: #0f172a; color: white; margin: 10px 0; box-sizing: border-box; }

            /* Results */
            .result-box { margin-top: 20px; padding: 15px; background: #0f172a; border-radius: 8px; display: none; }
            .confidence-bar { height: 6px; background: #334155; margin-top: 10px; border-radius: 3px; overflow: hidden; }
            .confidence-fill { height: 100%; background: var(--green); width: 0%; transition: width 0.5s; }
            
            /* Table */
            table { width: 100%; border-collapse: collapse; margin-top: 15px; }
            th, td { text-align: left; padding: 12px; border-bottom: 1px solid #334155; }
            .status-correct { color: var(--green); }
            .status-wrong { color: #ef4444; }
        </style>
    </head>
    <body>
        <div class="container">
            <div class="tabs">
                <button class="tab-btn active" onclick="showTab('verify')">1:1 Verify</button>
                <button class="tab-btn" onclick="showTab('search')">1:N Search DB</button>
                <button class="tab-btn" onclick="showTab('dashboard')">Dashboard & Stats</button>
            </div>

            <div id="verify" class="tab-content active">
                <div class="card">
                    <h2>👥 Verify Two Faces</h2>
                    <div style="display: grid; grid-template-columns: 1fr 1fr; gap: 15px;">
                        <div class="file-box" onclick="document.getElementById('v1').click()">Select Face 1<input type="file" id="v1" hidden></div>
                        <div class="file-box" onclick="document.getElementById('v2').click()">Select Face 2<input type="file" id="v2" hidden></div>
                    </div>
                    <button class="action" onclick="runVerify()">Compare Faces</button>
                    
                    <div id="verify-res" class="result-box">
                        <h3 id="v-text">Match Found</h3>
                        <p>Confidence: <span id="v-score">0%</span></p>
                        <div class="confidence-bar"><div id="v-bar" class="confidence-fill"></div></div>
                    </div>
                </div>
            </div>

            <div id="search" class="tab-content">
                <div class="card">
                    <h2>🔍 Search Database</h2>
                    <div class="file-box" onclick="document.getElementById('s1').click()">Select Face to Search<input type="file" id="s1" hidden></div>
                    <button class="action" onclick="runSearch()">Identify Person</button>
                    
                    <div id="search-res" class="result-box">
                        <h3 id="s-text">Unknown</h3>
                        <p>Confidence: <span id="s-score">0%</span></p>
                        <hr style="border-color: #334155; margin: 15px 0;">
                        <p style="font-size: 0.9rem; color: #94a3b8;">Is this person new? Add them to DB:</p>
                        <input type="text" id="new-name" placeholder="Enter Name">
                        <button class="action" style="background: #475569;" onclick="registerFace()">Register as New Person</button>
                    </div>
                </div>
            </div>

            <div id="dashboard" class="tab-content">
                <div class="card">
                    <h2>📊 System Stats</h2>
                    <div style="display:flex; justify-content:space-between; text-align:center; margin-bottom:20px;">
                        <div><h1 id="stat-acc">0%</h1><span>Accuracy</span></div>
                        <div><h1 id="stat-total">0</h1><span>Total Scans</span></div>
                    </div>
                    <h3>Recent History</h3>
                    <table id="history-table">
                        <thead><tr><th>Time</th><th>Type</th><th>Result</th><th>Feedback</th></tr></thead>
                        <tbody></tbody>
                    </table>
                </div>
            </div>
        </div>

        <script>
            function showTab(id) {
                document.querySelectorAll('.tab-content').forEach(d => d.classList.remove('active'));
                document.querySelectorAll('.tab-btn').forEach(b => b.classList.remove('active'));
                document.getElementById(id).classList.add('active');
                event.target.classList.add('active');
                if(id === 'dashboard') loadHistory();
            }

            async function runVerify() {
                const f1 = document.getElementById('v1').files[0];
                const f2 = document.getElementById('v2').files[0];
                if(!f1 || !f2) return alert("Select 2 images");
                
                const fd = new FormData();
                fd.append("file1", f1); fd.append("file2", f2);
                
                document.querySelector('#verify .action').innerText = "Processing...";
                const res = await fetch('/api/verify', {method:'POST', body:fd});
                const data = await res.json();
                
                document.getElementById('verify-res').style.display = 'block';
                document.getElementById('v-text').innerText = data.match ? "✅ MATCH" : "⛔ NO MATCH";
                document.getElementById('v-score').innerText = (data.confidence*100).toFixed(1) + "%";
                document.getElementById('v-bar').style.width = (data.confidence*100) + "%";
                document.querySelector('#verify .action').innerText = "Compare Faces";
            }

            async function runSearch() {
                const f1 = document.getElementById('s1').files[0];
                if(!f1) return alert("Select 1 image");
                
                const fd = new FormData();
                fd.append("file", f1);
                
                document.querySelector('#search .action').innerText = "Searching...";
                const res = await fetch('/api/identify', {method:'POST', body:fd});
                const data = await res.json();
                
                document.getElementById('search-res').style.display = 'block';
                document.getElementById('s-text').innerText = "👤 " + data.name;
                document.getElementById('s-score').innerText = (data.confidence*100).toFixed(1) + "%";
                document.querySelector('#search .action').innerText = "Identify Person";
            }

            async function registerFace() {
                const f1 = document.getElementById('s1').files[0];
                const name = document.getElementById('new-name').value;
                if(!f1 || !name) return alert("Need Image and Name");
                
                const fd = new FormData();
                fd.append("file", f1);
                fd.append("name", name);
                
                const res = await fetch('/api/register', {method:'POST', body:fd});
                const data = await res.json();
                alert(data.message);
            }

            async function loadHistory() {
                const res = await fetch('/api/history');
                const data = await res.json();
                const tbody = document.querySelector("#history-table tbody");
                tbody.innerHTML = "";
                
                let correct = 0, total = 0;
                
                data.forEach(row => {
                    if(row.feedback !== 'pending') total++;
                    if(row.feedback === 'correct') correct++;
                    
                    const tr = document.createElement("tr");
                    let fb = row.feedback === 'pending' ? 
                        `<button onclick="sendFeedback(${row.id}, 'correct')">✅</button> <button onclick="sendFeedback(${row.id}, 'wrong')">❌</button>` : 
                        `<span class="status-${row.feedback}">${row.feedback.toUpperCase()}</span>`;
                    
                    tr.innerHTML = `<td>${row.timestamp}</td><td>${row.type}</td><td>${row.result}</td><td>${fb}</td>`;
                    tbody.appendChild(tr);
                });
                
                const acc = total > 0 ? (correct/total*100).toFixed(1) : 0;
                document.getElementById('stat-acc').innerText = acc + "%";
                document.getElementById('stat-total').innerText = data.length;
            }

            async function sendFeedback(id, type) {
                await fetch('/api/feedback', {
                    method:'POST', headers:{'Content-Type':'application/json'},
                    body: JSON.stringify({id, type})
                });
                loadHistory();
            }
        </script>
    </body>
    </html>
    """
    
    class Feedback(BaseModel):
        id: int
        type: str

    @web.get("/")
    def index():
        return HTMLResponse(content=HTML_CONTENT)

    @web.post("/api/verify")
    async def verify_endpoint(file1: UploadFile = File(...), file2: UploadFile = File(...)):
        b1 = await file1.read()
        b2 = await file2.read()
        
        # Call GPU
        emb1 = FaceEngine().get_embedding_remote.remote(b1)
        emb2 = FaceEngine().get_embedding_remote.remote(b2)
        
        if emb1 is None or emb2 is None: return {"error": "Face not detected"}
        
        score = np.dot(emb1, emb2) # Already normalized
        match = bool(score > 0.6)
        
        # Log
        ts = datetime.now().strftime("%H:%M:%S")
        c.execute("INSERT INTO logs (type, result, confidence, feedback, timestamp) VALUES (?, ?, ?, ?, ?)",
                  ("1:1 Verify", "Match" if match else "No Match", float(score), "pending", ts))
        conn.commit()
        
        return {"match": match, "confidence": float(score)}

    @web.post("/api/identify")
    async def identify_endpoint(file: UploadFile = File(...)):
        b = await file.read()
        name, score = FaceEngine().search_face.remote(b)
        
        ts = datetime.now().strftime("%H:%M:%S")
        c.execute("INSERT INTO logs (type, result, confidence, feedback, timestamp) VALUES (?, ?, ?, ?, ?)",
                  ("1:N Search", name, float(score), "pending", ts))
        conn.commit()
        
        return {"name": name, "confidence": score}

    @web.post("/api/register")
    async def register_endpoint(file: UploadFile = File(...), name: str = Form(...)):
        b = await file.read()
        success, msg = FaceEngine().register_new_face.remote(b, name)
        return {"success": success, "message": msg}

    @web.get("/api/history")
    def history_endpoint():
        c.execute("SELECT * FROM logs ORDER BY id DESC LIMIT 50")
        rows = c.fetchall()
        return [{"id":r[0], "type":r[1], "result":r[2], "confidence":r[3], "feedback":r[4], "timestamp":r[5]} for r in rows]

    @web.post("/api/feedback")
    def feedback_endpoint(data: Feedback):
        c.execute("UPDATE logs SET feedback = ? WHERE id = ?", (data.type, data.id))
        conn.commit()
        return {"status": "ok"}

    return web