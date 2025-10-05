import os
import torch
import torch.nn as nn
from torchvision import transforms, models
from torch.utils.data import DataLoader, Dataset
from PIL import Image
import numpy as np
from tqdm import tqdm

# ---------- CONFIG ----------
DATA_ROOT = "/content/drive/MyDrive/celebdf_frames_test"  # or your local test path
CHECKPOINT_PATH = "/content/drive/MyDrive/deepfake-detection-videos/checkpoints/finetuned_images.pth"
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Device:", DEVICE)

# ---------- TRANSFORM ----------
transform = transforms.Compose([
    transforms.Resize((224,224)),
    transforms.ToTensor(),
    transforms.Normalize([0.485,0.456,0.406],[0.229,0.224,0.225])
])

# ---------- CUSTOM DATASET ----------
class CelebDFDataset(Dataset):
    def __init__(self, root, transform=None):
        self.samples = []
        self.transform = transform
        for label, cls in enumerate(["real", "fake"]):
            folder = os.path.join(root, cls)
            if not os.path.exists(folder): continue
            for fname in os.listdir(folder):
                if fname.endswith(".jpg"):
                    self.samples.append((os.path.join(folder, fname), label))
        print(f"Loaded {len(self.samples)} frames total.")
    
    def __len__(self):
        return len(self.samples)
    
    def __getitem__(self, idx):
        path, label = self.samples[idx]
        img = Image.open(path).convert("RGB")
        if self.transform:
            img = self.transform(img)
        return img, label

# ---------- LOAD MODEL ----------
model = models.efficientnet_b0(weights=None)
in_features = model.classifier[1].in_features
model.classifier[1] = nn.Linear(in_features, 2)
model.load_state_dict(torch.load(CHECKPOINT_PATH, map_location=DEVICE))
model = model.to(DEVICE)
model.eval()
print("✅ Model loaded successfully.")

# ---------- LOAD DATA ----------
dataset = CelebDFDataset(DATA_ROOT, transform)
loader = DataLoader(dataset, batch_size=64, shuffle=False)

# ---------- EVALUATE ----------
correct, total = 0, 0
with torch.no_grad():
    for imgs, labels in tqdm(loader):
        imgs, labels = imgs.to(DEVICE), labels.to(DEVICE)
        outs = model(imgs)
        preds = outs.argmax(1)
        correct += (preds == labels).sum().item()
        total += labels.size(0)

acc = correct / total
print(f"\n✅ Frame-level accuracy: {acc:.4f}")