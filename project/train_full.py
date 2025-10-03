import os, cv2, torch, time
import torch.nn as nn
import torch.optim as optim
from torchvision import transforms, models
from torch.utils.data import Dataset, DataLoader
from PIL import Image

# --------------------------
# Dataset loader with timeout
# --------------------------
class VideoFrameDataset(Dataset):
    def __init__(self, root, transform=None, max_frames=20, timeout=5):
        self.samples = []
        self.transform = transform
        self.max_frames = max_frames
        self.timeout = timeout
        classes = {"real":0, "fake":1}
        for label in classes:
            folder = os.path.join(root, label)
            for file in os.listdir(folder):
                if file.endswith(".mp4"):
                    self.samples.append((os.path.join(folder, file), classes[label]))

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        video_path, label = self.samples[idx]
        frames = []
        try:
            start = time.time()
            cap = cv2.VideoCapture(video_path)
            count = 0
            while True:
                if time.time() - start > self.timeout:
                    print(f"[WARN] Timeout reading {video_path}, skipping")
                    break
                ret, frame = cap.read()
                if not ret or count >= self.max_frames:
                    break
                if self.transform:
                    img = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
                    frames.append(self.transform(img))
                count += 1
            cap.release()
        except Exception as e:
            print(f"[WARN] Error reading {video_path}: {e}")

        # If no frames were read, return dummy tensor
        if len(frames) == 0:
            frames = [torch.zeros(3,224,224)]
        video_tensor = torch.stack(frames).mean(0)  # average frames
        return video_tensor, label


# --------------------------
# Training
# --------------------------
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Device:", device)

transform = transforms.Compose([
    transforms.Resize((224,224)),
    transforms.ToTensor(),
    transforms.Normalize([0.485,0.456,0.406],[0.229,0.224,0.225])
])

dataset = VideoFrameDataset("/content/drive/MyDrive/celebdf", transform=transform, max_frames=20)
train_loader = DataLoader(dataset, batch_size=8, shuffle=True, num_workers=2)

model = models.efficientnet_b0(weights=models.EfficientNet_B0_Weights.IMAGENET1K_V1)
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