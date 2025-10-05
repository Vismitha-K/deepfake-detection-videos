import os, sys, argparse, subprocess, shutil
import numpy as np
import cv2
from PIL import Image
import torch
from torchvision import transforms, models
import torch.nn.functional as F

print(">>> Grad-CAM demo script started <<<")

# ----------------------------------------------------
# Ensure local imports work when running this file
# ----------------------------------------------------
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)


def preprocess_frame(img_bgr, size=299):
    img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
    img = Image.fromarray(img_rgb)
    tf = transforms.Compose([
        transforms.Resize((size, size)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485,0.456,0.406], std=[0.229,0.224,0.225])
    ])
    return tf(img).unsqueeze(0)


def extract_frames(video_path, every_n=5, max_frames=300):
    cap = cv2.VideoCapture(video_path)
    frames = []
    idx, saved = 0, 0
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


def save_index_html(folder, thumbnails, title="Grad-CAM results"):
    html = f"<html><body><h2>{title}</h2>"
    for t in thumbnails:
        html += f"<div style='display:inline-block;margin:6px;text-align:center;'>"
        html += f"<img src='{os.path.basename(t)}' width=240><br>{os.path.basename(t)}</div>"
    html += "</body></html>"
    with open(os.path.join(folder, "index.html"), "w") as f:
        f.write(html)


# ----------------------------------------------------
# Video compression utilities (rarely needed)
# ----------------------------------------------------
def compress_with_ffmpeg(input_path, output_path, crf=28, preset="veryfast"):
    cmd = ["ffmpeg", "-y", "-i", input_path, "-vcodec", "libx264", "-crf", str(crf),
           "-preset", preset, "-acodec", "aac", "-b:a", "64k", output_path]
    print("Running:", " ".join(cmd))
    subprocess.run(cmd, check=True)


def compress_with_opencv(input_path, output_path, scale=0.6, fps_scale=1.0):
    cap = cv2.VideoCapture(input_path)
    if not cap.isOpened():
        raise RuntimeError("Cannot open video for compression fallback.")
    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH) * scale)
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT) * scale)
    fps = cap.get(cv2.CAP_PROP_FPS) * fps_scale
    fourcc = cv2.VideoWriter_fourcc(*'XVID')
    out = cv2.VideoWriter(output_path, fourcc, max(1, int(fps)), (w, h))
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        small = cv2.resize(frame, (w, h))
        out.write(small)
    cap.release()
    out.release()


# ----------------------------------------------------
# Main Grad-CAM analysis
# ----------------------------------------------------
def main(args):
    device = torch.device("cuda" if (args.use_cuda and torch.cuda.is_available()) else "cpu")
    print("Device:", device)

    # ------------------------------------------------
    # Load fine-tuned EfficientNet-B0 model
    # ------------------------------------------------
    print("Using finetuned EfficientNet-B0 model from checkpoint.")
    model = models.efficientnet_b0(weights=None)
    in_features = model.classifier[1].in_features
    model.classifier[1] = torch.nn.Linear(in_features, 2)

    # Adjust path depending on environment
    checkpoint_path = "checkpoints/finetuned_images.pth"
    if not os.path.exists(checkpoint_path):
        checkpoint_path = "/content/drive/MyDrive/checkpoints/finetuned_images.pth"

    state_dict = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(state_dict)
    model.to(device).eval()

    # ------------------------------------------------
    # Initialize Grad-CAM
    # ------------------------------------------------
    from gradcam import GradCAM, overlay_cam_on_image
    gradcam = GradCAM(model, use_cuda=(device.type == "cuda"))

    # ------------------------------------------------
    # Per-video analysis
    # ------------------------------------------------
    def run_analysis(video_path, out_dir_suffix):
        frames = extract_frames(video_path, every_n=args.every_n, max_frames=args.max_frames)
        print(f"Extracted {len(frames)} frames from {video_path}")

        results_folder = os.path.join(args.out, os.path.splitext(os.path.basename(video_path))[0] + out_dir_suffix)
        os.makedirs(results_folder, exist_ok=True)

        per_frame = []
        for i, (frame_index, frame_bgr) in enumerate(frames):
            inp = preprocess_frame(frame_bgr, size=args.input_size).to(device)
            with torch.no_grad():
                logits = model(inp)
                probs = F.softmax(logits, dim=1).cpu().numpy()[0] if logits.dim() > 1 else np.array([1.0, 0.0])
            num_classes = probs.shape[0]
            prob_fake = float(probs[1]) if num_classes > 1 else float(probs.max())
            cam = gradcam.generate_cam(inp, target_class=1 if num_classes > 1 else None)
            cam_energy = float(cam.mean())
            score = prob_fake + args.alpha * cam_energy
            overlay = overlay_cam_on_image(frame_bgr, cam, alpha=args.alpha_overlay)

            out_name = f"{i:03d}_frame{frame_index}_p{prob_fake:.3f}_ce{cam_energy:.4f}.jpg"
            out_path = os.path.join(results_folder, out_name)
            cv2.imwrite(out_path, overlay)
            per_frame.append((out_path, prob_fake, cam_energy, score))

        sorted_frames = sorted(per_frame, key=lambda x: x[3], reverse=True)
        topk = sorted_frames[:args.topk]
        thumbs = [p[0] for p in topk]
        save_index_html(results_folder, thumbs, title=f"Grad-CAM results ({out_dir_suffix.strip('_')})")

        avg_prob = float(np.mean([p[1] for p in per_frame])) if per_frame else 0.0
        with open(os.path.join(results_folder, "report.txt"), "w") as f:
            f.write(f"video: {video_path}\nframes_analyzed: {len(per_frame)}\navg_prob_fake: {avg_prob:.4f}\nTopK:\n")
            for p in topk:
                f.write(f"{os.path.basename(p[0])}\tprob_fake={p[1]:.4f}\tcam_energy={p[2]:.4f}\tscore={p[3]:.4f}\n")

        print("Saved results to:", results_folder)
        return results_folder

    orig_folder = run_analysis(args.video, "_orig")
    print("--- Finished. ---")
    print("Open these files in browser to view results:")
    print(orig_folder + "/index.html")


# ----------------------------------------------------
# CLI
# ----------------------------------------------------
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--video", required=True, help="input video path")
    parser.add_argument("--out", default="results", help="output folder")
    parser.add_argument("--every-n", type=int, default=5, help="sample every n-th frame")
    parser.add_argument("--max-frames", type=int, default=200, help="max frames to analyze")
    parser.add_argument("--topk", type=int, default=5, help="top K frames to show")
    parser.add_argument("--input-size", type=int, default=299, help="model input size")
    parser.add_argument("--alpha", type=float, default=1.0, help="weight for cam energy when ranking")
    parser.add_argument("--alpha-overlay", type=float, default=0.5, help="overlay transparency")
    parser.add_argument("--use-cuda", action="store_true", help="use cuda if available")
    parser.add_argument("--compress", action="store_true", help="compress video before analysis")
    parser.add_argument("--crf", type=int, default=28, help="CRF value for ffmpeg compression")
    parser.add_argument("--preset", type=str, default="veryfast", help="ffmpeg preset")
    parser.add_argument("--scale", type=float, default=0.6, help="scale factor for OpenCV compression fallback")
    parser.add_argument("--fps-scale", type=float, default=1.0, help="fps scaling for compression fallback")
    args = parser.parse_args()
    main(args)