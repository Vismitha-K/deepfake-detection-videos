# project/evaluate_multi.py
import os, json, argparse, csv
import torch
import torch.nn.functional as F
from torchvision import datasets, transforms, models
from tqdm import tqdm
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
import numpy as np

# reuse get_model from train_multi (or duplicate a minimal factory)
def get_model(name, num_classes=2):
    name = name.lower()
    if name == "resnet50":
        m = models.resnet50(weights=None)
        m.fc = torch.nn.Linear(m.fc.in_features, num_classes)
    elif name == "densenet121":
        m = models.densenet121(weights=None)
        m.classifier = torch.nn.Linear(m.classifier.in_features, num_classes)
    elif name == "mobilenet_v3_large":
        m = models.mobilenet_v3_large(weights=None)
        m.classifier = torch.nn.Sequential(torch.nn.Dropout(0.2), torch.nn.Linear(m.classifier[3].in_features, num_classes))
    elif name == "efficientnet_b0":
        m = models.efficientnet_b0(weights=None)
        m.classifier[1] = torch.nn.Linear(m.classifier[1].in_features, num_classes)
    else:
        raise ValueError(name)
    return m

def evaluate_model(model_name, ckpt_path, data_root, out_dir, device):
    os.makedirs(out_dir, exist_ok=True)
    # transforms
    transform = transforms.Compose([
        transforms.Resize((224,224)),
        transforms.ToTensor(),
        transforms.Normalize([0.485,0.456,0.406],[0.229,0.224,0.225])
    ])
    dataset = datasets.ImageFolder(data_root, transform=transform)
    loader = torch.utils.data.DataLoader(dataset, batch_size=64, shuffle=False, num_workers=2)
    print(f"Loaded {len(dataset)} frames")

    model = get_model(model_name)
    state = torch.load(ckpt_path, map_location=device)
    model.load_state_dict(state)
    model.to(device).eval()

    all_labels, all_preds, all_probs, filepaths = [], [], [], []
    # file path list from dataset
    idx_to_path = [p[0] for p in dataset.imgs]

    with torch.no_grad():
        for i, (imgs, labels) in enumerate(tqdm(loader)):
            imgs = imgs.to(device)
            outs = model(imgs)
            probs = F.softmax(outs, dim=1).cpu().numpy()
            preds = probs.argmax(axis=1)
            start = i*loader.batch_size
            for j in range(len(preds)):
                idx = start + j
                if idx >= len(idx_to_path): break
                filepaths.append(idx_to_path[idx])
            all_labels.extend(labels.numpy().tolist())
            all_preds.extend(preds.tolist())
            all_probs.extend(probs.tolist())

    # metrics
    acc = accuracy_score(all_labels, all_preds)
    prec = precision_score(all_labels, all_preds, zero_division=0)
    rec = recall_score(all_labels, all_preds, zero_division=0)
    f1 = f1_score(all_labels, all_preds, zero_division=0)
    metrics = {"accuracy": acc, "precision": prec, "recall": rec, "f1": f1, "n": len(all_labels)}
    print(model_name, metrics)

    # save per-frame CSV
    csv_path = os.path.join(out_dir, f"{model_name}_per_frame.csv")
    with open(csv_path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["filepath","label","pred","prob0","prob1"])
        for fp, lab, pred, prob in zip(filepaths, all_labels, all_preds, all_probs):
            w.writerow([fp, lab, pred, prob[0], prob[1]])

    # save metrics json
    with open(os.path.join(out_dir, f"{model_name}_metrics.json"), "w") as f:
        json.dump(metrics, f, indent=2)

    return metrics, csv_path

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--models", type=str, default="resnet50,densenet121,mobilenet_v3_large,efficientnet_b0")
    parser.add_argument("--ckpt-dir", required=True)
    parser.add_argument("--data-root", required=True)
    parser.add_argument("--out-dir", default="./multi_eval")
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    os.makedirs(args.out_dir, exist_ok=True)
    model_names = [m.strip() for m in args.models.split(",")]

    summary = {}
    for m in model_names:
        ckpt = os.path.join(args.ckpt_dir, f"{m}_best.pth")
        metrics, csv_path = evaluate_model(m, ckpt, args.data_root, args.out_dir, device)
        summary[m] = {"metrics": metrics, "per_frame_csv": csv_path}

    with open(os.path.join(args.out_dir, "multi_metrics_summary.json"), "w") as f:
        json.dump(summary, f, indent=2)

    print("Evaluation complete. Summary saved.")