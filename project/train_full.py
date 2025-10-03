import os, cv2, torch
import torch.nn as nn
import torch.optim as optim
from torchvision import transforms, models
from torch.utils.data import Dataset, DataLoader
from PIL import Image

# --------------------------
# Dataset loader (safe skip)
# --------------------------
class VideoFrameDataset(Dataset):
    def __init__(self, root, transform=None, max_frames=30, log_file="bad_videos.txt"):
        self.samples = []
        self.transform = transform
        self.max_frames = max_frames
        self.log_file = log_file
        classes = {"real": 0, "fake": 1}

        for label in classes:
            folder = os.path.join(root, label)
            if not os.path.exists(folder):
                continue
            for file in os.listdir(folder):
                if file.endswith(".mp4"):
                    self.samples.append((os.path.join(folder, file), classes[label]))

        # clear old log
        open(self.log_file, "w").close()

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        video_path, label = self.samples[idx]
        cap = cv2.VideoCapture(video_path)

        if not cap.isOpened():
            self.log_bad(video_path, "cannot open")
            return torch.zeros(3, 224, 224), label

        frames, count = [], 0
        while True:
            ret, frame = cap.read()
            if not ret or count >= self.max_frames:
                break
            if self.transform:
                img = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
                frames.append(self.transform(img))
            count += 1
        cap.release()

        if len(frames) == 0:
            self.log_bad(video_path, "no frames")
            return torch.zeros(3, 224, 224), label

        video_tensor = torch.stack(frames).mean(0)  # average frames
        return video_tensor, label

    def log_bad(self, path, reason):
        with open(self.log_file, "a") as f:
            f.write(f"{path} - {reason}\n")
        print(f"⚠️ Skipping {path} ({reason})")


# --------------------------
# Training setup
# --------------------------
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Device:", device)

transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

DATASET_ROOT = "/content/drive/MyDrive/celebdf"  # adjust if different
dataset = VideoFrameDataset(DATASET_ROOT, transform=transform, max_frames=30)
train_loader = DataLoader(dataset, batch_size=8, shuffle=True, num_workers=2)

model = models.efficientnet_b0(weights="IMAGENET1K_V1")
in_features = model.classifier[1].in_features
model.classifier[1] = nn.Linear(in_features, 2)
model = model.to(device)

criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=1e-4)

EPOCHS = 3
for epoch in range(EPOCHS):
    model.train()
    total, correct = 0, 0
    for inputs, labels in train_loader:
        inputs, labels = inputs.to(device), labels.to(device)
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        _, preds = torch.max(outputs, 1)
        correct += (preds == labels).sum().item()
        total += labels.size(0)

    acc = 100 * correct / total
    print(f"Epoch {epoch+1}/{EPOCHS}, Accuracy: {acc:.2f}%")

os.makedirs("checkpoints", exist_ok=True)
torch.save(model.state_dict(), "checkpoints/finetuned_full.pth")
print("✅ Model saved at checkpoints/finetuned_full.pth")
print("⚠️ Bad videos logged in bad_videos.txt")