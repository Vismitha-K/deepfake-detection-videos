import os
import torch
import torch.nn.functional as F
from torchvision import datasets, transforms, models
from tqdm import tqdm
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
import json

# -----------------------------
# CONFIGURATION
# -----------------------------
DATA_ROOT = "/content/drive/MyDrive/celebdf_frames"  # path to your frames folder
MODEL_PATH = "/content/drive/MyDrive/deepfake-detection-videos/checkpoints/finetuned_images.pth"
OUT_DIR = "/content/drive/MyDrive/deepfake-detection-videos/results_eval_frames"

os.makedirs(OUT_DIR, exist_ok=True)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Device: {device}")

# -----------------------------
# LOAD MODEL
# -----------------------------
from torchvision.models import efficientnet_b0

model = efficientnet_b0(weights=None)
in_features = model.classifier[1].in_features
model.classifier[1] = torch.nn.Linear(in_features, 2)

state_dict = torch.load(MODEL_PATH, map_location=device)
model.load_state_dict(state_dict)
model.to(device)
model.eval()

print("✅ Model loaded successfully from:", MODEL_PATH)

# -----------------------------
# DATASET SETUP
# -----------------------------
transform = transforms.Compose([
    transforms.Resize((224,224)),
    transforms.ToTensor(),
    transforms.Normalize([0.485,0.456,0.406], [0.229,0.224,0.225])
])

dataset = datasets.ImageFolder(DATA_ROOT, transform=transform)
loader = torch.utils.data.DataLoader(dataset, batch_size=64, shuffle=False, num_workers=2)

print(f"Loaded {len(dataset)} frames from {DATA_ROOT}")

# -----------------------------
# EVALUATION LOOP
# -----------------------------
all_labels, all_preds = [], []

with torch.no_grad():
    for imgs, labels in tqdm(loader, desc="Evaluating frames"):
        imgs, labels = imgs.to(device), labels.to(device)
        outputs = model(imgs)
        preds = torch.argmax(F.softmax(outputs, dim=1), dim=1)
        all_labels.extend(labels.cpu().numpy())
        all_preds.extend(preds.cpu().numpy())

# -----------------------------
# METRICS CALCULATION
# -----------------------------
acc = accuracy_score(all_labels, all_preds)
prec = precision_score(all_labels, all_preds)
rec = recall_score(all_labels, all_preds)
f1 = f1_score(all_labels, all_preds)

summary = {
    "total_frames": len(dataset),
    "accuracy": round(acc, 4),
    "precision": round(prec, 4),
    "recall": round(rec, 4),
    "f1": round(f1, 4)
}

print("\n✅ Evaluation complete!")
print(f"Frames evaluated: {summary['total_frames']}")
print(f"Accuracy:  {summary['accuracy']*100:.2f}%")
print(f"Precision: {summary['precision']*100:.2f}%")
print(f"Recall:    {summary['recall']*100:.2f}%")
print(f"F1 Score:  {summary['f1']*100:.2f}%")

# -----------------------------
# SAVE RESULTS
# -----------------------------
metrics_path = os.path.join(OUT_DIR, "metrics_summary.txt")
json_path = os.path.join(OUT_DIR, "metrics_summary.json")

with open(metrics_path, "w") as f:
    for k, v in summary.items():
        f.write(f"{k}: {v}\n")

with open(json_path, "w") as jf:
    json.dump(summary, jf, indent=4)

print(f"\n📁 Metrics saved to: {metrics_path}")