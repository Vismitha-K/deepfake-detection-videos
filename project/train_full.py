import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, random_split
from torchvision import datasets, transforms, models

# ✅ Config
DATASET_ROOT = "/content/drive/MyDrive/celebdf"  # change if needed
CHECKPOINT_DIR = "checkpoints"
EPOCHS = 15
BATCH_SIZE = 16
LR = 1e-4
VAL_SPLIT = 0.15

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Device:", device)

# ✅ Create transform
transform = transforms.Compose([
    transforms.Resize((299, 299)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485,0.456,0.406], std=[0.229,0.224,0.225])
])

# ✅ Load dataset
full_dataset = datasets.ImageFolder(DATASET_ROOT, transform=transform)

# Train/val split
val_size = int(len(full_dataset) * VAL_SPLIT)
train_size = len(full_dataset) - val_size
train_dataset, val_dataset = random_split(full_dataset, [train_size, val_size])

train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=2)
val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=2)

# ✅ Model (EfficientNet-B0 fine-tuned)
model = models.efficientnet_b0(weights=models.EfficientNet_B0_Weights.DEFAULT)
in_features = model.classifier[1].in_features
model.classifier[1] = nn.Linear(in_features, 2)  # binary classification
model = model.to(device)

criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=LR)

# ✅ Training loop
os.makedirs(CHECKPOINT_DIR, exist_ok=True)
best_val_acc = 0.0

for epoch in range(EPOCHS):
    model.train()
    correct, total, train_loss = 0, 0, 0.0
    for imgs, labels in train_loader:
        imgs, labels = imgs.to(device), labels.to(device)
        optimizer.zero_grad()
        outputs = model(imgs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        
        train_loss += loss.item()
        _, predicted = outputs.max(1)
        total += labels.size(0)
        correct += predicted.eq(labels).sum().item()
    
    train_acc = 100. * correct / total
    avg_loss = train_loss / len(train_loader)

    # ✅ Validation
    model.eval()
    val_correct, val_total = 0, 0
    with torch.no_grad():
        for imgs, labels in val_loader:
            imgs, labels = imgs.to(device), labels.to(device)
            outputs = model(imgs)
            _, predicted = outputs.max(1)
            val_total += labels.size(0)
            val_correct += predicted.eq(labels).sum().item()
    val_acc = 100. * val_correct / val_total

    print(f"Epoch {epoch+1}/{EPOCHS}, Train Loss: {avg_loss:.4f}, Train Acc: {train_acc:.2f}%, Val Acc: {val_acc:.2f}%")

    # Save best model
    if val_acc > best_val_acc:
        best_val_acc = val_acc
        torch.save(model.state_dict(), os.path.join(CHECKPOINT_DIR, "finetuned_full.pth"))
        print(f"✅ Saved best model (val acc {val_acc:.2f}%)")

print("Training complete. Best Val Acc:", best_val_acc)