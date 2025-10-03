import os, cv2, random, torch
import torch.nn as nn
import torch.optim as optim
from torchvision import transforms, models
from torch.utils.data import Dataset, DataLoader
from PIL import Image

# --------------------------
# Dataset loader
# --------------------------
class VideoFrameDataset(Dataset):
    def __init__(self, root, transform=None, frames_per_video=10, max_frames=100):
        self.samples = []
        self.transform = transform
        self.frames_per_video = frames_per_video
        self.max_frames = max_frames
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
        cap = cv2.VideoCapture(video_path)
        frames = []
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

        # Pick random frame indices
        if total_frames > 0:
            frame_indices = sorted(random.sample(
                range(min(total_frames, self.max_frames)), 
                min(self.frames_per_video, total_frames)
            ))
        else:
            frame_indices = []

        count, selected = 0, []
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            if count in frame_indices:
                img = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
                if self.transform:
                    img = self.transform(img)
                selected.append(img)
            count += 1
        cap.release()

        if len(selected) == 0:
            selected = [torch.zeros(3,224,224)]

        # stack → average frames to 1 tensor
        video_tensor = torch.stack(selected).mean(0)
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

# Change path to full dataset root
DATASET_ROOT = "/content/drive/MyDrive/celebdf"  
dataset = VideoFrameDataset(DATASET_ROOT, transform=transform, frames_per_video=10)
train_loader = DataLoader(dataset, batch_size=8, shuffle=True, num_workers=2)

model = models.efficientnet_b0(pretrained=True)
in_features = model.classifier[1].in_features
model.classifier[1] = nn.Linear(in_features, 2)
model = model.to(device)

criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=1e-4)

EPOCHS = 5  # can increase if GPU allows
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

# Save model
os.makedirs("checkpoints", exist_ok=True)
torch.save(model.state_dict(), "checkpoints/finetuned_full.pth")
print("✅ Model saved at checkpoints/finetuned_full.pth")