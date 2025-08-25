# train_rgbd_with_objects_transformer_attention.py
import os
from pathlib import Path
from typing import List

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
from PIL import Image
import cv2
import numpy as np
from tqdm.auto import tqdm
from accelerate import Accelerator
from ultralytics import YOLO

# -----------------------------
# 0. Utils
# -----------------------------
def set_seed(seed: int = 42):
    import random
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def list_images_recursively(root: str | Path) -> List[str]:
    exts = (".jpg", ".jpeg", ".png", ".JPG", ".PNG", ".bmp")
    paths = []
    root = Path(root)
    for p in root.rglob("*"):
        if p.suffix in exts:
            paths.append(str(p))
    return sorted(paths)


# -----------------------------
# 1. Dataset (RGB + Depth)
# -----------------------------
class RGBDDataset(Dataset):
    def __init__(self, rgb_dir, depth_dir, transform_rgb=None, transform_depth=None, img_size=(224, 224)):
        self.rgb_dir = Path(rgb_dir)
        self.depth_dir = Path(depth_dir)
        self.transform_rgb = transform_rgb
        self.transform_depth = transform_depth
        self.img_size = img_size

        self.samples = []
        self.class_to_idx = {}
        for idx, class_dir in enumerate(sorted(self.rgb_dir.iterdir())):
            if class_dir.is_dir():
                self.class_to_idx[class_dir.name] = idx
                for img_path in class_dir.glob("*.[jp][pn]g"):
                    depth_path = self.depth_dir / class_dir.name / f"{img_path.stem}_depth.png"
                    if depth_path.exists():
                        self.samples.append((img_path, depth_path, idx))

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        rgb_path, depth_path, label = self.samples[idx]
        rgb = cv2.imread(str(rgb_path))[:, :, ::-1]
        rgb = cv2.resize(rgb, self.img_size)
        rgb = Image.fromarray(rgb)
        depth = cv2.imread(str(depth_path), cv2.IMREAD_GRAYSCALE)
        depth = cv2.resize(depth, self.img_size)
        depth = Image.fromarray(depth)

        if self.transform_rgb:
            rgb = self.transform_rgb(rgb)
        else:
            rgb = transforms.ToTensor()(rgb)
        if self.transform_depth:
            depth = self.transform_depth(depth)
        else:
            depth = transforms.ToTensor()(depth)
        depth = depth.repeat(3, 1, 1)

        return rgb, depth, str(rgb_path), label


# -----------------------------
# 2. Transformer-style Cross-Attention Block
# -----------------------------
class CrossAttentionBlock(nn.Module):
    def __init__(self, embed_dim, num_heads=8, ffn_hidden=2048, dropout=0.1):
        super().__init__()
        self.attn_rgb2depth = nn.MultiheadAttention(embed_dim=embed_dim, num_heads=num_heads, batch_first=True)
        self.attn_depth2rgb = nn.MultiheadAttention(embed_dim=embed_dim, num_heads=num_heads, batch_first=True)

        self.norm_rgb = nn.LayerNorm(embed_dim)
        self.norm_depth = nn.LayerNorm(embed_dim)

        self.ffn_rgb = nn.Sequential(
            nn.Linear(embed_dim, ffn_hidden),
            nn.GELU(),
            nn.Linear(ffn_hidden, embed_dim),
            nn.Dropout(dropout)
        )
        self.ffn_depth = nn.Sequential(
            nn.Linear(embed_dim, ffn_hidden),
            nn.GELU(),
            nn.Linear(ffn_hidden, embed_dim),
            nn.Dropout(dropout)
        )

    def forward(self, rgb_tokens, depth_tokens):
        # Cross-Attention
        rgb_attn, _ = self.attn_rgb2depth(query=rgb_tokens, key=depth_tokens, value=depth_tokens)
        depth_attn, _ = self.attn_depth2rgb(query=depth_tokens, key=rgb_tokens, value=rgb_tokens)

        # Residual + Norm + FFN
        rgb_tokens = self.norm_rgb(rgb_tokens + rgb_attn)
        rgb_tokens = rgb_tokens + self.ffn_rgb(rgb_tokens)

        depth_tokens = self.norm_depth(depth_tokens + depth_attn)
        depth_tokens = depth_tokens + self.ffn_depth(depth_tokens)

        return rgb_tokens, depth_tokens


# -----------------------------
# 3. MultiModalFusionHead
# -----------------------------
class MultiModalFusionHead(nn.Module):
    def __init__(self, rgb_backbone, depth_backbone, object_dim, num_classes=365,
                 layers=4, attn_dim=512, num_heads=8, cross_layers=2):
        super().__init__()
        self.rgb_backbone = rgb_backbone
        self.depth_backbone = depth_backbone
        self.layers = layers

        embed_dim = rgb_backbone.embed_dim

        # Cross-Attention Blocks
        self.cross_blocks = nn.ModuleList([
            CrossAttentionBlock(embed_dim=embed_dim, num_heads=num_heads) for _ in range(cross_layers)
        ])

        # Projection
        self.fc_proj_rgb = nn.Linear(embed_dim, attn_dim)
        self.fc_proj_depth = nn.Linear(embed_dim, attn_dim)

        self.input_dim = attn_dim * 2 + object_dim
        self.head = nn.Sequential(
            nn.Linear(self.input_dim, 1024),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),
            nn.Linear(1024, num_classes)
        )

    def forward(self, rgb, depth, obj_feat):
        # Feature extraction
        if self.layers == 1:
            x_rgb = self.rgb_backbone.forward_features(rgb)
            rgb_patch = x_rgb["x_norm_patchtokens"]
            rgb_cls = x_rgb["x_norm_clstoken"][:, None, :]
            x_depth = self.depth_backbone.forward_features(depth)
            depth_patch = x_depth["x_norm_patchtokens"]
            depth_cls = x_depth["x_norm_clstoken"][:, None, :]
        else:
            x_rgb = self.rgb_backbone.get_intermediate_layers(rgb, n=4, return_class_token=True)
            rgb_patch = x_rgb[3][0]
            rgb_cls = x_rgb[3][1][:, None, :]
            x_depth = self.depth_backbone.get_intermediate_layers(depth, n=4, return_class_token=True)
            depth_patch = x_depth[3][0]
            depth_cls = x_depth[3][1][:, None, :]

        rgb_tokens = torch.cat([rgb_cls, rgb_patch], dim=1)
        depth_tokens = torch.cat([depth_cls, depth_patch], dim=1)

        # Apply stacked cross-attention blocks
        for block in self.cross_blocks:
            rgb_tokens, depth_tokens = block(rgb_tokens, depth_tokens)

        # Pooling + projection
        rgb_feat = self.fc_proj_rgb(rgb_tokens.mean(dim=1))
        depth_feat = self.fc_proj_depth(depth_tokens.mean(dim=1))

        # Concatenate with object features
        fused = torch.cat([rgb_feat, depth_feat, obj_feat], dim=1)
        return self.head(fused)


# -----------------------------
# 4. YOLO Feature Extractor (on-the-fly)
# -----------------------------
@torch.no_grad()
def yolo_features(yolo_model, img_paths, top_k=5, device="cuda"):
    results = yolo_model(img_paths, verbose=False, device=device)
    nc = yolo_model.model.nc
    feats = []
    for r in results:
        scores = []
        if hasattr(r, "boxes") and r.boxes is not None and len(r.boxes) > 0:
            cls_ids = r.boxes.cls.detach().cpu().numpy().astype(int)
            confs = r.boxes.conf.detach().cpu().numpy()
            for cid, cf in zip(cls_ids, confs):
                one_hot = np.zeros(nc, dtype=np.float32)
                one_hot[cid] = float(cf)
                scores.append(one_hot)
        if len(scores) == 0:
            feats.append(torch.zeros(nc, dtype=torch.float32, device=device))
        else:
            scores = np.stack(scores, axis=0)
            if scores.shape[0] > top_k:
                idx = np.argsort(-scores.max(axis=1))[:top_k]
                scores = scores[idx]
            feats.append(torch.tensor(scores.mean(axis=0), dtype=torch.float32, device=device))
    return torch.stack(feats, dim=0)  # [B, nc]


# -----------------------------
# 5. Train / Eval
# -----------------------------
def train_one_epoch(model, loader, criterion, optimizer, accelerator, yolo_model, top_k=5):
    model.train()
    total_loss, correct, total = 0.0, 0, 0
    pbar = tqdm(loader, disable=not accelerator.is_local_main_process)
    for rgb, depth, paths, labels in pbar:
        rgb, depth, labels = rgb.to(accelerator.device), depth.to(accelerator.device), labels.to(accelerator.device)
        obj_feat = yolo_features(yolo_model, list(paths), top_k=top_k, device=accelerator.device)

        optimizer.zero_grad()
        outputs = model(rgb, depth, obj_feat)
        loss = criterion(outputs, labels)
        accelerator.backward(loss)
        optimizer.step()

        preds = outputs.argmax(1)
        batch_correct = (preds == labels).sum().item()
        total_loss += loss.item() * rgb.size(0)
        correct += batch_correct
        total += labels.size(0)

        pbar.set_postfix({
            "avg_loss": f"{total_loss/total:.4f}",
            "avg_acc": f"{correct/total:.4f}"
        })
    return total_loss / total, correct / total


@torch.no_grad()
def evaluate(model, loader, criterion, accelerator, yolo_model, top_k=5):
    model.eval()
    total_loss, correct, total = 0.0, 0, 0
    for rgb, depth, paths, labels in tqdm(loader, disable=not accelerator.is_local_main_process):
        rgb, depth, labels = rgb.to(accelerator.device), depth.to(accelerator.device), labels.to(accelerator.device)
        obj_feat = yolo_features(yolo_model, list(paths), top_k=top_k, device=accelerator.device)
        outputs = model(rgb, depth, obj_feat)
        loss = criterion(outputs, labels)
        preds = outputs.argmax(1)
        total_loss += loss.item() * rgb.size(0)
        correct += (preds == labels).sum().item()
        total += labels.size(0)
    return total_loss / total, correct / total


# -----------------------------
# 6. Main
# -----------------------------
def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--rgb_train_dir", type=str, default="/4TB_2nd/Seongmin/scene/train")
    parser.add_argument("--rgb_val_dir", type=str, default="/4TB_2nd/Seongmin/scene/val")
    parser.add_argument("--depth_train_dir", type=str, default="/4TB_2nd/Seongmin/scene_depth/train")
    parser.add_argument("--depth_val_dir", type=str, default="/4TB_2nd/Seongmin/scene_depth/val")
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--num_workers", type=int, default=8)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--layers", type=int, default=4, choices=[1, 4])
    parser.add_argument("--yolo_model", type=str, default="/home/sysadmin/Seongmin/SSV2A_Mul/weights/SSV2A/yolov8x-oi7.pt")
    parser.add_argument("--top_k", type=int, default=3)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    set_seed(args.seed)
    accelerator = Accelerator()
    device = accelerator.device

    tfms_rgb = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.RandomHorizontalFlip(),
        transforms.ColorJitter(0.2, 0.2, 0.2, 0.05),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406],
                             [0.229, 0.224, 0.225]),
    ])
    tfms_depth = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize([0.5], [0.5]),
    ])

    train_dataset = RGBDDataset(args.rgb_train_dir, args.depth_train_dir, transform_rgb=tfms_rgb, transform_depth=tfms_depth)
    val_dataset = RGBDDataset(args.rgb_val_dir, args.depth_val_dir, transform_rgb=tfms_rgb, transform_depth=tfms_depth)

    num_classes = len(train_dataset.class_to_idx)
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True,
                              num_workers=args.num_workers, pin_memory=True)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False,
                            num_workers=args.num_workers, pin_memory=True)

    accelerator.print("[Backbone] Loading DINOv2 vitl14_reg (RGB/Depth)")
    rgb_backbone = torch.hub.load("facebookresearch/dinov2", "dinov2_vitl14_reg")
    depth_backbone = torch.hub.load("facebookresearch/dinov2", "dinov2_vitl14_reg")
    for p in rgb_backbone.parameters():
        p.requires_grad = False
    for p in depth_backbone.parameters():
        p.requires_grad = False

    accelerator.print(f"[YOLO] Loading {args.yolo_model}")
    yolo_model = YOLO(args.yolo_model).to(device)
    object_dim = yolo_model.model.nc

    model = MultiModalFusionHead(rgb_backbone, depth_backbone, object_dim=object_dim,
                                 num_classes=num_classes)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.AdamW(model.head.parameters(), lr=args.lr)

    model, optimizer, train_loader, val_loader = accelerator.prepare(model, optimizer, train_loader, val_loader)

    best_val_acc = 0.0
    for epoch in range(1, args.epochs + 1):
        train_loss, train_acc = train_one_epoch(model, train_loader, criterion, optimizer, accelerator, yolo_model, args.top_k)
        val_loss, val_acc = evaluate(model, val_loader, criterion, accelerator, yolo_model, args.top_k)

        accelerator.print(f"[Epoch {epoch:02d}] Train {train_loss:.4f}/{train_acc:.4f} | Val {val_loss:.4f}/{val_acc:.4f}")
        if accelerator.is_local_main_process and val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save(accelerator.unwrap_model(model).state_dict(), "best_attention.pth")
            accelerator.print(f"[CKPT] Saved best_attention.pth (val_acc={best_val_acc:.4f})")


if __name__ == "__main__":
    main()
