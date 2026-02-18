import streamlit as st
import modal
import numpy as np
import io
import asyncio

# --- CONFIGURATION ---
st.set_page_config(page_title="ViT Face AI", layout="wide")

# Connect to Modal App
# We use .lookup to find the running app on Modal's servers
try:
    f = modal.Cls.lookup("face-vit-backend", "FaceModel")
except Exception as e:
    st.error("Could not connect to Modal backend. Make sure you have deployed it!")
    st.stop()

st.title("🧠 Vision Transformer Face System")

# --- SIDEBAR ---
st.sidebar.header("Settings")
threshold = st.sidebar.slider("Match Threshold", 0.0, 1.0, 0.6)

# --- TABS ---
tab1, tab2 = st.tabs(["👥 Verify Faces", "⚙️ Model Manager"])

# === TAB 1: VERIFICATION ===
with tab1:
    st.header("Compare Two Faces")
    col1, col2 = st.columns(2)
    
    file1 = col1.file_uploader("Face 1", type=['jpg', 'png', 'jpeg'])
    file2 = col2.file_uploader("Face 2", type=['jpg', 'png', 'jpeg'])

    if st.button("RUN COMPARISON", type="primary"):
        if file1 and file2:
            with st.spinner("Processing on Cloud GPU..."):
                # Run both requests in parallel using Async
                async def run_inference():
                    task1 = f.get_embedding.remote.aio(file1.getvalue())
                    task2 = f.get_embedding.remote.aio(file2.getvalue())
                    return await asyncio.gather(task1, task2)
                
                emb1, emb2 = asyncio.run(run_inference())
                
                # Check results
                if emb1 is None or emb2 is None:
                    st.error("❌ Could not detect a face in one of the images.")
                else:
                    # Calculate Cosine Similarity
                    score = np.dot(emb1, emb2) / (np.linalg.norm(emb1) * np.linalg.norm(emb2))
                    
                    st.metric("Similarity Score", f"{score:.4f}")
                    
                    if score > threshold:
                        st.success("✅ MATCH CONFIRMED")
                    else:
                        st.error("⛔ NO MATCH")
        else:
            st.warning("Please upload both images.")

# === TAB 2: MODEL MANAGER ===
with tab2:
    st.header("Upload Fine-Tuned Weights")
    st.info("Upload your local .pth file here. It will be sent to the Cloud Volume.")
    
    weights_file = st.file_uploader("Choose model file (.pth)", type=["pth", "pt"])
    
    if weights_file:
        if st.button("⬆️ Upload to Cloud Volume"):
            with st.spinner("Uploading large file... this may take a minute..."):
                try:
                    # Call the remote save function
                    f.save_custom_weights.remote(weights_file.getvalue())
                    st.success("✅ Upload Complete! The model has been updated.")
                    st.info("The next time you run a comparison, the new weights will be loaded.")
                except Exception as e:
                    st.error(f"Upload failed: {e}")