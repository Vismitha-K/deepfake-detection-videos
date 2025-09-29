import os, sys, argparse, shutil, subprocess
import numpy as np
import cv2
from PIL import Image
import torch
import torch.nn.functional as F
from torchvision import transforms, models
from tqdm import tqdm

# add project dir to sys.path so gradcam.py can be imported
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)

# import GradCAM implemented earlier
from gradcam import GradCAM

def try_import_repo_model():
    possible = [
        ("src.model", "xception"),
        ("model", "xception"),
        ("models", "xception"),
        ("src.models", "xception"),
        ("src.model.xception", None),
        ("model.xception", None),
    ]
    for pkg, attr in possible:
        try:
            mod = __import__(pkg, fromlist=['*'])
            if attr:
                cls = getattr(mod, attr, None)
                if cls is None:
                    cls = getattr(mod, 'Xception', None) or getattr(mod, 'xception', None)
                if cls:
                    try:
                        return cls(num_classes=2)
                    except Exception:
                        try:
                            return cls()
                        except Exception:
                            continue
            else:
                cls = getattr(mod, 'Xception', None) or getattr(mod, 'xception', None)
                if cls:
                    try:
                        return cls(num_classes=2)
                    except Exception:
                        try:
                            return cls()
                        except Exception:
                            continue
        except Exception:
            continue
    return None

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
    idx = 0
    saved = 0
    while True:
        ret, frame = cap.read()
        if not ret: break
        if idx % every_n == 0:
            frames.append((idx, frame.copy()))
            saved += 1
            if saved >= max_frames: break
        idx += 1
    cap.release()
    return frames

def compress_video_ffmpeg(in_path, out_path, crf=28, preset="veryfast"):
    cmd = ["ffmpeg","-y","-i", in_path, "-vcodec","libx264","-crf", str(crf), "-preset", preset, "-acodec", "aac", "-b:a", "64k", out_path]
    subprocess.run(cmd, check=True)

def analyze_video(model, gradcam, video_path, device, args):
    frames = extract_frames(video_path, every_n=args.every_n, max_frames=args.max_frames)
    per_frame = []
    for idx, frame in frames:
        inp = preprocess_frame(frame, size=args.input_size).to(device)
        with torch.no_grad():
            logits = model(inp)
            if logits.dim() > 1:
                probs = F.softmax(logits, dim=1).cpu().numpy()[0]
            else:
                probs = np.array([1.0, 0.0])
        num_classes = probs.shape[0]
        prob_fake = float(probs[1]) if num_classes > 1 else float(probs.max())
        # compute grad-cam heatmap for fake class (needs backward inside)
        cam = gradcam.generate_cam(inp, target_class=1 if num_classes>1 else None)
        cam_energy = float(cam.mean())
        score = prob_fake + args.alpha * cam_energy
        per_frame.append({"prob": prob_fake, "cam_energy": cam_energy, "score": score})
    # compute baseline avg
    if len(per_frame) == 0:
        return None
    avg_prob = float(np.mean([p["prob"] for p in per_frame]))
    # pick top-K by score
    sorted_frames = sorted(per_frame, key=lambda x: x["score"], reverse=True)
    topk = sorted_frames[:args.topk] if args.topk > 0 else sorted_frames
    topk_avg = float(np.mean([p["prob"] for p in topk]))
    # majority vote (optional)
    majority_vote = 1 if np.mean([1 if p["prob"]>0.5 else 0 for p in per_frame]) >= 0.5 else 0
    return {
        "avg_prob": avg_prob,
        "topk_avg_prob": topk_avg,
        "frames_count": len(per_frame),
        "majority_vote": majority_vote
    }

def load_video_list(dataset_root=None, manifest=None):
    videos = []
    if manifest:
        import pandas as pd
        df = pd.read_csv(manifest)
        for _, r in df.iterrows():
            p = r['path']
            if not os.path.isabs(p):
                p = os.path.join(os.path.dirname(manifest), p)
            label = r['label']
            if isinstance(label, str):
                label = 1 if label.lower() in ('fake','1','true','t') else 0
            videos.append((p, int(label)))
        return videos
    # else expect dataset_root/real and dataset_root/fake
    real_dir = os.path.join(dataset_root, "real")
    fake_dir = os.path.join(dataset_root, "fake")
    if not os.path.isdir(real_dir) or not os.path.isdir(fake_dir):
        raise FileNotFoundError("Expecting folders: dataset_root/real and dataset_root/fake")
    for f in sorted(os.listdir(real_dir)):
        if f.lower().endswith(('.mp4','.mov','.avi','.mkv')):
            videos.append((os.path.join(real_dir,f), 0))
    for f in sorted(os.listdir(fake_dir)):
        if f.lower().endswith(('.mp4','.mov','.avi','.mkv')):
            videos.append((os.path.join(fake_dir,f), 1))
    return videos

def compute_metrics(gt, pred):
    try:
        from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
        return {
            "accuracy": float(accuracy_score(gt,pred)),
            "precision": float(precision_score(gt,pred, zero_division=0)),
            "recall": float(recall_score(gt,pred, zero_division=0)),
            "f1": float(f1_score(gt,pred, zero_division=0))
        }
    except Exception:
        # simple fallback
        gt = np.array(gt); pred = np.array(pred)
        tp = int(((gt==1) & (pred==1)).sum())
        tn = int(((gt==0) & (pred==0)).sum())
        fp = int(((gt==0) & (pred==1)).sum())
        fn = int(((gt==1) & (pred==0)).sum())
        acc = (tp+tn) / max(1, (tp+tn+fp+fn))
        prec = tp / (tp+fp) if (tp+fp)>0 else 0.0
        rec = tp / (tp+fn) if (tp+fn)>0 else 0.0
        f1 = 2*prec*rec/(prec+rec) if (prec+rec)>0 else 0.0
        return {"accuracy":acc,"precision":prec,"recall":rec,"f1":f1}

def main(args):
    device = torch.device("cuda" if (args.use_cuda and torch.cuda.is_available()) else "cpu")
    print("Device:", device)
    # load model
    model = try_import_repo_model()
    if model is None:
        print("Using torchvision EfficientNet-B0 fallback.")
        model = models.efficientnet_b0(pretrained=True)
        in_features = model.classifier[1].in_features
        model.classifier = torch.nn.Sequential(torch.nn.Dropout(0.2), torch.nn.Linear(in_features, 2))
    model.to(device).eval()
    # try load checkpoint if supplied
    if args.model_path:
        if os.path.exists(args.model_path):
            ck = torch.load(args.model_path, map_location=device)
            if isinstance(ck, dict) and 'state_dict' in ck:
                sd = ck['state_dict']
                try:
                    model.load_state_dict(sd)
                    print("Loaded state_dict")
                except Exception:
                    model.load_state_dict({k.replace('module.',''):v for k,v in sd.items()})
            else:
                try:
                    model.load_state_dict(ck)
                except Exception:
                    model = ck
                    model.to(device).eval()

    gradcam = GradCAM(model, use_cuda=(device.type=="cuda"))

    # load video list
    videos = load_video_list(args.dataset_root, args.manifest)
    print("Videos to evaluate:", len(videos))

    results = []
    for vid_path, label in tqdm(videos, desc="Videos"):
        # original
        orig_res = analyze_video(model, gradcam, vid_path, device, args)
        if orig_res is None:
            print("Skipping (no frames):", vid_path)
            continue
        # optionally compressed
        comp_res = None
        if args.compress:
            base, ext = os.path.splitext(vid_path)
            comp_path = base + "_eval_compressed.mp4"
            try:
                if shutil.which("ffmpeg"):
                    compress_video_ffmpeg(vid_path, comp_path, crf=args.crf, preset=args.preset)
                else:
                    # simple opencv downscale fallback
                    cap = cv2.VideoCapture(vid_path)
                    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH) * args.scale)
                    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT) * args.scale)
                    fps = cap.get(cv2.CAP_PROP_FPS) or 10
                    fourcc = cv2.VideoWriter_fourcc(*'XVID')
                    out = cv2.VideoWriter(comp_path, fourcc, max(1,int(fps)), (w,h))
                    while True:
                        ret, fr = cap.read()
                        if not ret: break
                        out.write(cv2.resize(fr,(w,h)))
                    cap.release(); out.release()
                comp_res = analyze_video(model, gradcam, comp_path, device, args)
            except Exception as e:
                print("Compression failed for", vid_path, e)
        results.append({
            "video": vid_path,
            "label": label,
            "orig_avg": orig_res["avg_prob"],
            "orig_topk_avg": orig_res["topk_avg_prob"],
            "orig_frames": orig_res["frames_count"],
            "comp_avg": comp_res["avg_prob"] if comp_res else None,
            "comp_topk_avg": comp_res["topk_avg_prob"] if comp_res else None
        })

    # compute predictions and metrics
    gt = [r["label"] for r in results]
    # baseline preds = orig_avg > thresh
    preds_baseline = [1 if r["orig_avg"] > args.thresh else 0 for r in results]
    preds_topk = [1 if r["orig_topk_avg"] > args.thresh else 0 for r in results]

    m_baseline = compute_metrics(gt, preds_baseline)
    m_topk = compute_metrics(gt, preds_topk)

    summary = {
        "n": len(results),
        "baseline": m_baseline,
        "topk": m_topk
    }

    # save detailed CSV
    import csv
    out_csv = os.path.join(args.out, "per_video_results.csv")
    os.makedirs(args.out, exist_ok=True)
    with open(out_csv, "w", newline="") as f:
        w = csv.writer(f)
        header = ["video","label","orig_avg","orig_topk_avg","orig_frames","comp_avg","comp_topk_avg"]
        w.writerow(header)
        for r in results:
            w.writerow([r.get(h) for h in header])

    with open(os.path.join(args.out, "metrics_summary.txt"), "w") as f:
        f.write(str(summary))

    print("DONE. Summary:")
    print(summary)
    print("Per-video CSV saved at:", out_csv)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset-root", required=False, help="path to dataset root with /real and /fake subfolders")
    parser.add_argument("--manifest", required=False, help="csv manifest with columns path,label (label: fake/real or 1/0)")
    parser.add_argument("--out", default="results_eval", help="output folder")
    parser.add_argument("--every-n", type=int, default=6, help="sample every n-th frame")
    parser.add_argument("--max-frames", type=int, default=300, help="max frames per video")
    parser.add_argument("--topk", type=int, default=5, help="number of top suspicious frames to use for Top-K average")
    parser.add_argument("--input-size", type=int, default=299, help="model input size")
    parser.add_argument("--alpha", type=float, default=1.0, help="weight for cam energy when scoring frames")
    parser.add_argument("--thresh", type=float, default=0.5, help="threshold on avg prob to declare FAKE")
    parser.add_argument("--use-cuda", action="store_true", help="use GPU if available")
    parser.add_argument("--model-path", default="", help="path to .pth checkpoint")
    parser.add_argument("--compress", action="store_true", help="also evaluate on compressed copy")
    parser.add_argument("--crf", type=int, default=28, help="ffmpeg crf for compression")
    parser.add_argument("--preset", type=str, default="veryfast", help="ffmpeg preset")
    parser.add_argument("--scale", type=float, default=0.6, help="scale fallback for compression")
    args = parser.parse_args()
    if not args.dataset_root and not args.manifest:
        parser.error("Provide --dataset-root or --manifest")
    main(args)