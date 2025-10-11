import os, random, json, argparse, shutil, subprocess
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms, models
from torch.utils.data import DataLoader
from PIL import Image, UnidentifiedImageError

# ==============================================================
#                 GOOGLE DRIVE SAFETY SETUP
# ==============================================================

USE_DRIVE = True   # set to False if running locally (non-Colab)

if USE_DRIVE:
    DRIVE_ROOT = "/content/drive/MyDrive/deepfake-detection-videos"
    DRIVE_DATA = "/content/drive/MyDrive/celebdf_frames"
    LOCAL_DATA = "/content/celebdf_frames"

    def dataset_incomplete(path):
        if not os.path.exists(path):
            return True
        size_mb = int(subprocess.getoutput(f"du -sm {path} | cut -f1"))
        return size_mb < 1000   # less than ~1 GB → incomplete copy

    if dataset_incomplete(LOCAL_DATA):
        print("⏳ Copying dataset from Drive to local SSD...")
        if os.path.exists(LOCAL_DATA):
            shutil.rmtree(LOCAL_DATA)
        shutil.copytree(DRIVE_DATA, LOCAL_DATA)
        print("✅ Dataset copied to local SSD.")
    else:
        print("✅ Local dataset already exists and looks complete, skipping copy.")

    DEFAULT_OUT_DIR = os.path.join(DRIVE_ROOT, "multi_results")
    DEFAULT_CKPT_DIR = os.path.join(DRIVE_ROOT, "checkpoints")
    os.makedirs(DEFAULT_OUT_DIR, exist_ok=True)
    os.makedirs(DEFAULT_CKPT_DIR, exist_ok=True)
else:
    LOCAL_DATA = "./data"
    DEFAULT_OUT_DIR = "./multi_results"
    DEFAULT_CKPT_DIR = "./checkpoints"
    os.makedirs(DEFAULT_OUT_DIR, exist_ok=True)
    os.makedirs(DEFAULT_CKPT_DIR, exist_ok=True)


# ==============================================================
#                 REPRODUCIBILITY
# ==============================================================
def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


# ==============================================================
#                 MODEL FACTORY
# ==============================================================
def get_model(name, num_classes=2, pretrained=True):
    name = name.lower()
    if name == "resnet50":
        m = models.resnet50(weights=models.ResNet50_Weights.IMAGENET1K_V2 if pretrained else None)
        m.fc = nn.Linear(m.fc.in_features, num_classes)
    elif name == "densenet121":
        m = models.densenet121(weights=models.DenseNet121_Weights.IMAGENET1K_V1 if pretrained else None)
        m.classifier = nn.Linear(m.classifier.in_features, num_classes)
    elif name == "mobilenet_v3_large":
        m = models.mobilenet_v3_large(weights=models.MobileNet_V3_Large_Weights.IMAGENET1K_V2 if pretrained else None)
        in_f = m.classifier[3].in_features if isinstance(m.classifier, nn.Sequential) else m.classifier.in_features
        m.classifier = nn.Sequential(nn.Dropout(0.2), nn.Linear(in_f, num_classes))
    elif name == "efficientnet_b0":
        m = models.efficientnet_b0(weights=models.EfficientNet_B0_Weights.IMAGENET1K_V1 if pretrained else None)
        m.classifier[1] = nn.Linear(m.classifier[1].in_features, num_classes)
    else:
        raise ValueError(f"Unsupported model: {name}")
    return m


# ==============================================================
#                 SAFE IMAGE LOADER
# ==============================================================
def safe_image_loader(path):
    try:
        with open(path, "rb") as f:
            img = Image.open(f)
            return img.convert("RGB")
    except (UnidentifiedImageError, OSError):
        print(f"⚠️ Skipping corrupt image: {path}")
        return Image.new("RGB", (224, 224), (0, 0, 0))


from torchvision.datasets import ImageFolder
class SafeImageFolder(ImageFolder):
    def __init__(self, root, transform=None):
        super().__init__(root, transform=transform)
        self.loader = safe_image_loader


# ==============================================================
#                 TRAINING LOOP
# ==============================================================
def train_model(model_name, data_root, out_dir, device, epochs=8, batch_size=64, lr=1e-4, seed=42):
    set_seed(seed)
    os.makedirs(out_dir, exist_ok=True)
    print(f"\n=== Training {model_name} ===")

    # transforms
    transform_train = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])
    transform_eval = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])

    # --- VIDEO-LEVEL SPLIT ---
    full_folder = SafeImageFolder(data_root, transform=None)
    samples = full_folder.samples
    print(f"📂 Total frames found: {len(samples)}")

    from torch.utils.data import Dataset

    class FramesDataset(Dataset):
        def __init__(self, samples, transform=None, loader=None):
            self.samples = samples
            self.transform = transform
            self.loader = loader if loader is not None else full_folder.loader
        def __len__(self):
            return len(self.samples)
        def __getitem__(self, idx):
            path, label = self.samples[idx]
            img = self.loader(path)
            if self.transform:
                img = self.transform(img)
            return img, label

    # Group frames by video ID (filename prefix before last underscore)
    video_to_indices = {}
    for i, (p, lbl) in enumerate(samples):
        name = os.path.basename(p)
        vid = "_".join(name.split("_")[:-1]) if "_" in name else name
        video_to_indices.setdefault(vid, []).append(i)

    video_ids = list(video_to_indices.keys())
    random.shuffle(video_ids)
    n = len(video_ids)
    n_train, n_val = int(0.7 * n), int(0.15 * n)
    train_vids = video_ids[:n_train]
    val_vids = video_ids[n_train:n_train + n_val]
    test_vids = video_ids[n_train + n_val:]

    def collect(vlist):
        idx = []
        for v in vlist:
            idx.extend(video_to_indices[v])
        return idx

    train_idx, val_idx, test_idx = collect(train_vids), collect(val_vids), collect(test_vids)

    assert set(train_idx).isdisjoint(val_idx)
    assert set(train_idx).isdisjoint(test_idx)
    assert set(val_idx).isdisjoint(test_idx)

    train_dataset = FramesDataset([samples[i] for i in train_idx], transform=transform_train)
    val_dataset = FramesDataset([samples[i] for i in val_idx], transform=transform_eval)
    test_dataset = FramesDataset([samples[i] for i in test_idx], transform=transform_eval)
    print(f"📊 Video-level split → Train: {len(train_dataset)}, Val: {len(val_dataset)}, Test: {len(test_dataset)}")

    num_workers = 4 if data_root.startswith("/content/") else 0
    pin_memory = True if (device.type == "cuda" and data_root.startswith("/content/")) else False

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers, pin_memory=pin_memory)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers)

    # model setup
    model = get_model(model_name, num_classes=2, pretrained=True)
    model = model.to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=lr)

    best_val = 0.0
    history = {"train_acc": [], "val_acc": [], "train_loss": [], "val_loss": []}

    ckpt_dir = os.path.join(DEFAULT_CKPT_DIR, model_name)
    os.makedirs(ckpt_dir, exist_ok=True)
    model_ckpt_path = os.path.join(ckpt_dir, f"{model_name}_resume.pth")
    best_model_path = os.path.join(ckpt_dir, f"{model_name}_best.pth")
    history_path = os.path.join(out_dir, f"{model_name}_history.json")

    # Resume if interrupted
    if os.path.exists(model_ckpt_path):
        print(f"🔁 Resuming from checkpoint: {model_ckpt_path}")
        model.load_state_dict(torch.load(model_ckpt_path, map_location=device))

    for ep in range(epochs):
        model.train()
        correct, total, running_loss = 0, 0, 0.0
        for imgs, labels in train_loader:
            imgs, labels = imgs.to(device), labels.to(device)
            optimizer.zero_grad()
            outs = model(imgs)
            loss = criterion(outs, labels)
            loss.backward()
            optimizer.step()
            running_loss += loss.item() * imgs.size(0)
            _, preds = torch.max(outs, 1)
            correct += (preds == labels).sum().item()
            total += labels.size(0)
        train_acc = correct / total
        train_loss = running_loss / total

        # validation
        model.eval()
        vcorrect, vtotal, vloss = 0, 0, 0.0
        with torch.no_grad():
            for imgs, labels in val_loader:
                imgs, labels = imgs.to(device), labels.to(device)
                outs = model(imgs)
                loss = criterion(outs, labels)
                vloss += loss.item() * imgs.size(0)
                _, preds = torch.max(outs, 1)
                vcorrect += (preds == labels).sum().item()
                vtotal += labels.size(0)
        val_acc = vcorrect / vtotal
        val_loss = vloss / vtotal

        history["train_acc"].append(train_acc)
        history["val_acc"].append(val_acc)
        history["train_loss"].append(train_loss)
        history["val_loss"].append(val_loss)

        print(f"[{model_name}] Epoch {ep+1}/{epochs}  train_acc={train_acc:.4f}  val_acc={val_acc:.4f}")

        # Always save progress
        torch.save(model.state_dict(), model_ckpt_path)
        if val_acc > best_val:
            best_val = val_acc
            torch.save(model.state_dict(), best_model_path)
            print(f"🏆 New best model saved to {best_model_path}")

        with open(history_path, "w") as f:
            json.dump(history, f, indent=2)

    print(f"✅ Finished training {model_name}. Best val acc = {best_val:.4f}")
    return best_model_path, history_path


# ==============================================================
#                 MAIN CLI
# ==============================================================
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--models", type=str, default="resnet50,densenet121,mobilenet_v3_large,efficientnet_b0")
    parser.add_argument("--data-root", type=str, default="/content/celebdf_frames")
    parser.add_argument("--out-dir", type=str, default=DEFAULT_OUT_DIR)
    parser.add_argument("--epochs", type=int, default=8)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    set_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Device:", device)

    models_list = [m.strip() for m in args.models.split(",") if m.strip()]
    summary = {}
    for mname in models_list:
        ckpt, hist = train_model(
            mname,
            args.data_root,
            args.out_dir,
            device,
            epochs=args.epochs,
            batch_size=args.batch_size,
            lr=args.lr,
            seed=args.seed
        )
        summary[mname] = {"ckpt": ckpt, "history": hist}

    with open(os.path.join(args.out_dir, "summary.json"), "w") as f:
        json.dump(summary, f, indent=2)
    print("✅ All training complete. Summary saved to Drive.")