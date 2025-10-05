import streamlit as st
import os
import torch
import torch.nn.functional as F
from torchvision import models, transforms
from PIL import Image
import numpy as np
import cv2
import time
from project.gradcam import GradCAM, overlay_cam_on_image

# -------------------------------
# Streamlit Page Setup
# -------------------------------
st.set_page_config(page_title="Deepfake Detection with Grad-CAM", layout="wide")
st.title("🎭 Deepfake Detection with Explainability")
st.write(
    "Upload a video, and our model will detect whether it is **REAL or FAKE** "
    "and highlight suspicious regions using Grad-CAM."
)

# -------------------------------
# Model Loading
# -------------------------------
@st.cache_resource
def load_model():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = models.efficientnet_b0(weights=None)
    in_features = model.classifier[1].in_features
    model.classifier[1] = torch.nn.Linear(in_features, 2)
    checkpoint_path = os.path.join("checkpoints", "finetuned_images.pth")
    model.load_state_dict(torch.load(checkpoint_path, map_location=device))
    model.to(device).eval()
    gradcam = GradCAM(model, use_cuda=(device.type == "cuda"))
    return model, gradcam, device

model, gradcam, device = load_model()

# -------------------------------
# Preprocessing utilities
# -------------------------------
def preprocess_frame(frame_bgr, size=299):
    img_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
    img = Image.fromarray(img_rgb)
    transform = transforms.Compose([
        transforms.Resize((size, size)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485,0.456,0.406], std=[0.229,0.224,0.225])
    ])
    return transform(img).unsqueeze(0)

def extract_frames(video_path, every_n=5, max_frames=200):
    cap = cv2.VideoCapture(video_path)
    frames = []
    idx = 0
    saved = 0
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        if idx % every_n == 0:
            frames.append((idx, frame.copy()))
            saved += 1
            if saved >= max_frames:
                break
        idx += 1
    cap.release()
    return frames

# -------------------------------
# Streamlit UI – Upload Video
# -------------------------------
uploaded_file = st.file_uploader("Upload a video file (mp4/avi)", type=["mp4", "avi"])

if uploaded_file:
    os.makedirs("uploads", exist_ok=True)
    os.makedirs("results", exist_ok=True)
    input_path = os.path.join("uploads", uploaded_file.name)
    with open(input_path, "wb") as f:
        f.write(uploaded_file.getbuffer())

    st.video(input_path)
    st.success("✅ Video uploaded successfully!")

    # -------------------------------
    # Analysis
    # -------------------------------
    with st.spinner("Analyzing video... please wait ⏳"):
        frames = extract_frames(input_path, every_n=6, max_frames=100)
        st.write(f"Extracted {len(frames)} frames for analysis.")
        per_frame_scores = []
        results_dir = os.path.join("results", os.path.splitext(uploaded_file.name)[0] + "_orig")
        os.makedirs(results_dir, exist_ok=True)

        for i, (frame_idx, frame) in enumerate(frames):
            inp = preprocess_frame(frame).to(device)
            with torch.no_grad():
                logits = model(inp)
                probs = F.softmax(logits, dim=1).cpu().numpy()[0]
                prob_fake = float(probs[1])
            cam = gradcam.generate_cam(inp, target_class=1)
            overlay = overlay_cam_on_image(frame, cam, alpha=0.5)
            out_name = f"{i:03d}_frame{frame_idx}_p{prob_fake:.3f}.jpg"
            out_path = os.path.join(results_dir, out_name)
            cv2.imwrite(out_path, overlay)
            per_frame_scores.append(prob_fake)

        avg_prob_fake = float(np.mean(per_frame_scores)) if per_frame_scores else 0.0
        verdict = "🟢 REAL" if avg_prob_fake < 0.5 else "🔴 FAKE"
        time.sleep(1)

    # -------------------------------
    # Display Results
    # -------------------------------
    st.subheader("📊 Detection Summary")
    st.markdown(f"### Final Verdict: **{verdict}**  (avg fake prob = {avg_prob_fake:.2f})")

    result_images = sorted([os.path.join(results_dir, f)
                            for f in os.listdir(results_dir) if f.endswith(".jpg")])[:5]
    if result_images:
        st.subheader("Top Frames with Grad-CAM Overlay")
        st.image(result_images, width=300, caption=[os.path.basename(i) for i in result_images])

    st.info("✅ Analysis complete. Results are saved in the 'results/' folder for future reference.")