import os, random 
import torch, torch.nn as nn, torch.optim as optim
from torchvision import datasets, transforms, models
from torch.utils.data import DataLoader, random_split
import json

# -----------------------------
# Permanent paths on Google Drive
# -----------------------------
DRIVE_ROOT = "/content/drive/MyDrive/deepfake-detection-videos"
DATA_ROOT = "/content/drive/MyDrive/celebdf_frames"
OUT_DIR = os.path.join(DRIVE_ROOT, "results_images")
CHECKPOINT_DIR = os.path.join(DRIVE_ROOT, "checkpoints")
os.makedirs(CHECKPOINT_DIR, exist_ok=True)
os.makedirs(OUT_DIR, exist_ok=True)

# -----------------------------
# Device setup
# -----------------------------
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Device:", device)

# -----------------------------
# Transform and dataset
# -----------------------------
transform = transforms.Compose([
    transforms.Resize((224,224)),
    transforms.ToTensor(),
    transforms.Normalize([0.485,0.456,0.406],[0.229,0.224,0.225])
])

dataset = datasets.ImageFolder(DATA_ROOT, transform=transform)
n = len(dataset)
print("Total images:", n)

# -----------------------------
# Split into train/val/test
# -----------------------------
n_train = int(0.7 * n)
n_val = int(0.15 * n)
n_test = n - n_train - n_val
train_set, val_set, test_set = random_split(dataset, [n_train, n_val, n_test])

batch_size = 64
train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True, num_workers=2)
val_loader = DataLoader(val_set, batch_size=batch_size, shuffle=False, num_workers=2)
test_loader = DataLoader(test_set, batch_size=batch_size, shuffle=False, num_workers=2)

# -----------------------------
# Model setup
# -----------------------------
from torchvision.models import efficientnet_b0
import torch.nn as nn
import torch

weights_path = "/content/drive/MyDrive/model_weights/efficientnet_b0_rwightman-7f5810bc.pth"

# Load pretrained EfficientNet-B0 weights from Drive
model = efficientnet_b0(weights=None)
state_dict = torch.load(weights_path, map_location=device)
model.load_state_dict(state_dict)

# Replace classifier head for binary classification (REAL vs FAKE)
in_features = model.classifier[1].in_features
model.classifier[1] = nn.Linear(in_features, 2)
model = model.to(device)

# Loss, optimizer, and training setup
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=1e-4)
epochs = 8
best_val = 0.0
history = {"train_acc": [], "val_acc": []}

# -----------------------------
# Training loop
# -----------------------------
for ep in range(epochs):
    model.train()
    correct, total = 0, 0
    for imgs, labels in train_loader:
        imgs, labels = imgs.to(device), labels.to(device)
        optimizer.zero_grad()
        outs = model(imgs)
        loss = criterion(outs, labels)
        loss.backward()
        optimizer.step()
        _, pred = outs.max(1)
        correct += (pred==labels).sum().item()
        total += labels.size(0)
    train_acc = correct/total

    # Validation
    model.eval()
    vcorrect, vtotal = 0, 0
    with torch.no_grad():
        for imgs, labels in val_loader:
            imgs, labels = imgs.to(device), labels.to(device)
            outs = model(imgs)
            _, pred = outs.max(1)
            vcorrect += (pred==labels).sum().item()
            vtotal += labels.size(0)
    val_acc = vcorrect / vtotal
    history["train_acc"].append(train_acc)
    history["val_acc"].append(val_acc)
    print(f"Epoch {ep+1}/{epochs} train_acc={train_acc:.4f} val_acc={val_acc:.4f}")

    # Save best model permanently
    if val_acc > best_val:
        best_val = val_acc
        torch.save(model.state_dict(), os.path.join(CHECKPOINT_DIR, "finetuned_images.pth"))
        print("Saved best model to Google Drive.")

# -----------------------------
# Test evaluation
# -----------------------------
model.load_state_dict(torch.load(os.path.join(CHECKPOINT_DIR, "finetuned_images.pth")))
model.eval()
tcorrect, ttotal = 0, 0
with torch.no_grad():
    for imgs, labels in test_loader:
        imgs, labels = imgs.to(device), labels.to(device)
        outs = model(imgs)
        _, pred = outs.max(1)
        tcorrect += (pred==labels).sum().item()
        ttotal += labels.size(0)
test_acc = tcorrect / ttotal
print("Test accuracy:", test_acc)

# -----------------------------
# Save results summary permanently
# -----------------------------
with open(os.path.join(OUT_DIR, "summary.json"), "w") as f:
    json.dump({"best_val": best_val, "test_acc": test_acc, "history": history}, f)
print("Done. Results saved to:", OUT_DIR)