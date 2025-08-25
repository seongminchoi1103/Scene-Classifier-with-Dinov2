import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
from pathlib import Path
from PIL import Image
import cv2
from tqdm import tqdm
from accelerate import Accelerator

# -----------------------------
# 1. Depth Precompute
# -----------------------------
def load_depth_model(device):
    from depth_anything_v2.dpt import DepthAnythingV2
    model = DepthAnythingV2(
        encoder='vitl',
        features=256,
        out_channels=[256, 512, 1024, 1024]
    )
    ckpt = "/home/sysadmin/Seongmin/SSV2A_Mul/Depth-Anything-V2/checkpoints/depth_anything_v2_vitl.pth"
    model.load_state_dict(torch.load(ckpt, map_location=device))
    return model.to(device).eval()

@torch.no_grad()
def precompute_depth(input_dir, output_dir, img_size=(224, 224), device="cpu"):
    model = load_depth_model(device)
    input_dir, output_dir = Path(input_dir), Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    wrongs = list(input_dir.rglob("*_depth*"))
    for p in wrongs:
        try: p.unlink()
        except: pass

    img_paths = list(input_dir.rglob("*.[jp][pn]g"))
    print(f"Found {len(img_paths)} images for depth precompute")

    for img_path in tqdm(img_paths, desc="Precomputing Depth"):
        rel_path = img_path.relative_to(input_dir)
        depth_path = (output_dir / rel_path).with_name(img_path.stem + "_depth.png")
        depth_path.parent.mkdir(parents=True, exist_ok=True)
        if depth_path.exists(): continue

        img_cv = cv2.imread(str(img_path))
        if img_cv is None: continue
        img_rgb = img_cv[:, :, ::-1]
        depth_map = model.infer_image(img_rgb)
        depth_resized = cv2.resize(depth_map, img_size, interpolation=cv2.INTER_LINEAR)
        dmin, dmax = depth_resized.min(), depth_resized.max()
        depth_norm = (depth_resized - dmin) / (dmax - dmin + 1e-6)
        depth_img = (depth_norm * 255).astype("uint8")
        cv2.imwrite(str(depth_path), depth_img)

# -----------------------------
# 2. Dataset
# -----------------------------
class RGBDDataset(Dataset):
    def __init__(self, rgb_dir, depth_dir, transform_rgb=None, transform_depth=None, img_size=(224, 224)):
        self.rgb_dir = Path(rgb_dir)
        self.depth_dir = Path(depth_dir)
        self.transform_rgb = transform_rgb
        self.transform_depth = transform_depth
        self.img_size = img_size
        self.samples, self.class_to_idx = [], {}
        for idx, class_dir in enumerate(sorted(self.rgb_dir.iterdir())):
            if class_dir.is_dir():
                self.class_to_idx[class_dir.name] = idx
                for img_path in class_dir.glob("*.[jp][pn]g"):
                    depth_path = self.depth_dir / class_dir.name / f"{img_path.stem}_depth.png"
                    if depth_path.exists():
                        self.samples.append((img_path, depth_path, idx))
    def __len__(self): return len(self.samples)
    def __getitem__(self, idx):
        rgb_path, depth_path, label = self.samples[idx]
        rgb = cv2.imread(str(rgb_path))[:, :, ::-1]
        rgb = cv2.resize(rgb, self.img_size)
        depth = cv2.imread(str(depth_path), cv2.IMREAD_GRAYSCALE)
        depth = cv2.resize(depth, self.img_size)
        rgb = Image.fromarray(rgb)
        depth = Image.fromarray(depth)
        rgb = self.transform_rgb(rgb) if self.transform_rgb else transforms.ToTensor()(rgb)
        depth = self.transform_depth(depth) if self.transform_depth else transforms.ToTensor()(depth)
        depth = depth.repeat(3, 1, 1)
        return rgb, depth, label

# -----------------------------
# 3. Model
# -----------------------------
class MultiLayerFusionHead(nn.Module):
    def __init__(self, rgb_backbone, depth_backbone, num_classes=365, layers=4):
        super().__init__()
        self.rgb_backbone = rgb_backbone
        self.depth_backbone = depth_backbone
        self.layers = layers
        embed_dim = rgb_backbone.embed_dim
        if layers == 4:
            self.input_dim = (layers + 1) * embed_dim * 2
        elif layers == 1:
            self.input_dim = 2 * embed_dim * 2
        else:
            raise ValueError("Only layers=1 or 4 supported")
        self.head = nn.Sequential(
            nn.Linear(self.input_dim, 1024),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(1024, num_classes),
        )
    def forward(self, rgb, depth):
        if self.layers == 1:
            x_rgb = self.rgb_backbone.forward_features(rgb)
            rgb_feat = torch.cat([x_rgb["x_norm_clstoken"], x_rgb["x_norm_patchtokens"].mean(dim=1)], dim=1)
            x_depth = self.depth_backbone.forward_features(depth)
            depth_feat = torch.cat([x_depth["x_norm_clstoken"], x_depth["x_norm_patchtokens"].mean(dim=1)], dim=1)
        else:
            x_rgb = self.rgb_backbone.get_intermediate_layers(rgb, n=4, return_class_token=True)
            rgb_feat = torch.cat([x_rgb[0][1], x_rgb[1][1], x_rgb[2][1], x_rgb[3][1], x_rgb[3][0].mean(dim=1)], dim=1)
            x_depth = self.depth_backbone.get_intermediate_layers(depth, n=4, return_class_token=True)
            depth_feat = torch.cat([x_depth[0][1], x_depth[1][1], x_depth[2][1], x_depth[3][1], x_depth[3][0].mean(dim=1)], dim=1)
        return self.head(torch.cat([rgb_feat, depth_feat], dim=1))

# -----------------------------
# 4. Train / Eval
# -----------------------------
def train_one_epoch(model, loader, criterion, optimizer, accelerator):
    model.train()
    total_loss, correct, total = 0, 0, 0
    pbar = tqdm(loader, desc="Training", leave=False)
    for batch in pbar:
        rgb, depth, labels = [x.to(accelerator.device) for x in batch]
        optimizer.zero_grad()
        outputs = model(rgb, depth)
        loss = criterion(outputs, labels)
        accelerator.backward(loss)
        optimizer.step()

        # -------------------------
        # ✅ batch 총합 loss 로 변환
        # -------------------------
        batch_sz = rgb.size(0)
        batch_loss = loss.detach() * batch_sz

        # gather (각 GPU에서 전체 합산)
        gathered_loss = accelerator.gather(batch_loss)     # per-sample 총합 loss
        gathered_labels = accelerator.gather(labels)
        gathered_preds = accelerator.gather(outputs.argmax(1))

        # per-batch metrics (각 GPU 합산 후 평균)
        batch_loss_mean = gathered_loss.sum().item() / gathered_labels.size(0)
        batch_acc = (gathered_preds == gathered_labels).float().mean().item()
        pbar.set_postfix({"loss": f"{batch_loss_mean:.4f}", "acc": f"{batch_acc:.4f}"})

        # accumulate for epoch (전체 dataset 단위로 합산)
        total_loss += gathered_loss.sum().item()
        correct += (gathered_preds == gathered_labels).sum().item()
        total += gathered_labels.size(0)

    avg_loss = total_loss / total
    avg_acc = correct / total
    return avg_loss, avg_acc


def evaluate(model, loader, criterion, accelerator):
    model.eval()
    total_loss, correct, total = 0, 0, 0
    with torch.no_grad():
        for batch in tqdm(loader, desc="Evaluating"):
            rgb, depth, labels = [x.to(accelerator.device) for x in batch]
            outputs = model(rgb, depth)
            loss = criterion(outputs, labels)

            # ✅ batch 총합 loss
            batch_sz = rgb.size(0)
            batch_loss = loss.detach() * batch_sz

            gathered_loss = accelerator.gather(batch_loss)
            gathered_labels = accelerator.gather(labels)
            gathered_preds = accelerator.gather(outputs.argmax(1))

            total_loss += gathered_loss.sum().item()
            correct += (gathered_preds == gathered_labels).sum().item()
            total += gathered_labels.size(0)

    return total_loss / total, correct / total


# -----------------------------
# 5. Main
# -----------------------------
if __name__ == "__main__":
    accelerator = Accelerator()
    device = accelerator.device

    # Transforms
    train_tfms_rgb = transforms.Compose([
        transforms.Resize((224,224)),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize([0.485,0.456,0.406],[0.229,0.224,0.225])
    ])
    train_tfms_depth = transforms.Compose([
        transforms.Resize((224,224)),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize([0.5],[0.5])
    ])

    rgb_train_dir = "/4TB_2nd/Seongmin/scene/train"
    depth_train_dir = "/4TB_2nd/Seongmin/scene_depth/train"
    rgb_val_dir = "/4TB_2nd/Seongmin/scene/val"
    depth_val_dir = "/4TB_2nd/Seongmin/scene_depth/val"

    train_dataset = RGBDDataset(rgb_train_dir, depth_train_dir,
                                transform_rgb=train_tfms_rgb, transform_depth=train_tfms_depth)
    val_dataset = RGBDDataset(rgb_val_dir, depth_val_dir,
                              transform_rgb=train_tfms_rgb, transform_depth=train_tfms_depth)

    train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True, num_workers=8)
    val_loader = DataLoader(val_dataset, batch_size=64, shuffle=False, num_workers=8)

    # Backbones
    rgb_backbone = torch.hub.load("facebookresearch/dinov2", "dinov2_vitl14_reg")
    depth_backbone = torch.hub.load("facebookresearch/dinov2", "dinov2_vitl14_reg")
    for p in rgb_backbone.parameters(): p.requires_grad = False
    for p in depth_backbone.parameters(): p.requires_grad = False

    model = MultiLayerFusionHead(rgb_backbone, depth_backbone,
                                 num_classes=len(train_dataset.class_to_idx), layers=4)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.AdamW(model.head.parameters(), lr=1e-4)

    # Prepare with Accelerate
    model, optimizer, train_loader, val_loader = accelerator.prepare(
        model, optimizer, train_loader, val_loader
    )

    # -----------------------------
    # Training Loop with best model saving
    # -----------------------------
    best_val_acc = 0.0

    for epoch in range(30):
        train_loss, train_acc = train_one_epoch(model, train_loader, criterion, optimizer, accelerator)
        val_loss, val_acc = evaluate(model, val_loader, criterion, accelerator)

        print(f"Epoch {epoch+1}: Train Loss {train_loss:.4f}, Acc {train_acc:.4f} | "
              f"Val Loss {val_loss:.4f}, Acc {val_acc:.4f}")

        # ✅ Save best model based on validation accuracy
        if accelerator.is_local_main_process and val_acc > best_val_acc:
            best_val_acc = val_acc
            accelerator.print(f"[CKPT] Saving best model (val_acc={best_val_acc:.4f})")
            torch.save(accelerator.unwrap_model(model).state_dict(), "best_model.pth")
