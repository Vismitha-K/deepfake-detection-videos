import streamlit as st
import os
import subprocess
import time
import sys

st.set_page_config(page_title="Deepfake Detection with Grad-CAM", layout="wide")

st.title("🎭 Deepfake Detection with Explainability")
st.write("Upload a video, and our model will detect whether it is **REAL or FAKE** and highlight suspicious regions using Grad-CAM.")

# Upload video
uploaded_file = st.file_uploader("Upload a video file (mp4/avi)", type=["mp4", "avi"])

if uploaded_file:
    # Save uploaded video
    input_path = os.path.join("uploads", uploaded_file.name)
    os.makedirs("uploads", exist_ok=True)
    with open(input_path, "wb") as f:
        f.write(uploaded_file.getbuffer())

    st.video(input_path)
    st.success("✅ Video uploaded successfully!")

    # Run gradcam_demo.py
    out_dir = os.path.join("results", os.path.splitext(uploaded_file.name)[0])
    cmd = [
        sys.executable, "project/gradcam_demo.py",
        "--video", input_path,
        "--out", "results",
        "--topk", "5",
        "--every-n", "6",
        "--input-size", "299"
    ]

    with st.spinner("Analyzing video... this may take a few minutes ⏳"):
        subprocess.run(cmd)
        time.sleep(2)

    # Display results
    result_folder = out_dir + "_orig"
    report_file = os.path.join(result_folder, "report.txt")
    index_file = os.path.join(result_folder, "index.html")

    if os.path.exists(report_file):
        st.subheader("📊 Detection Report")
        with open(report_file, "r") as f:
            report_text = f.read()
            st.text(report_text)

            # Extract avg_prob_fake from report
            avg_prob_line = [line for line in report_text.splitlines() if "avg_prob_fake" in line]
            if avg_prob_line:
                avg_prob = float(avg_prob_line[0].split(":")[-1])
                verdict = "🟢 REAL" if avg_prob < 0.5 else "🔴 FAKE"
                st.markdown(f"### Final Verdict: **{verdict}** (avg fake prob = {avg_prob:.2f})")


    if os.path.exists(result_folder):
        st.subheader("Top-K Suspicious Frames (Grad-CAM)")
        images = [os.path.join(result_folder, f) for f in os.listdir(result_folder) if f.endswith(".jpg")]
        images = sorted(images)[:5]  # show top 5
        st.image(images, width=300, caption=[os.path.basename(i) for i in images])

    st.info("Open `index.html` inside the results folder for a full visualization.")