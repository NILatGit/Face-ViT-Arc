import streamlit as st
import torch
import timm
from PIL import Image
from timm.data import create_transform

st.title("DINOv2 Image Classifier (112x112)")

# 1. Load Model
@st.cache_resource # Caches model so it doesn't reload on every click
def load_model():
    model_name = 'vit_base_patch14_dinov2.lvd142m'
    model = timm.create_model(model_name, pretrained=True, num_classes=1000)
    model.eval()
    return model

model = load_model()

# 2. Force 112x112 Preprocessing
# We override the default config to use your specific 112px requirement
transform = create_transform(
    input_size=(3, 112, 112),
    interpolation='bicubic',
    mean=(0.485, 0.456, 0.406),
    std=(0.229, 0.224, 0.225)
)

uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "png", "jpeg"])

if uploaded_file is not None:
    image = Image.open(uploaded_file).convert('RGB')
    st.image(image, caption='Uploaded Image', use_container_width=True)
    
    # Transform and Predict
    img_tensor = transform(image).unsqueeze(0)
    
    with torch.no_grad():
        logits = model(img_tensor)
        probs = torch.nn.functional.softmax(logits[0], dim=0)
    
    top_prob, top_idx = torch.topk(probs, 1)
    st.write(f"**Prediction Class ID:** {top_idx.item()}")
    st.write(f"**Confidence:** {top_prob.item():.2%}")