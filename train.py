import os
from pathlib import Path
from PIL import Image
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torch.utils.data.distributed import DistributedSampler
from torchvision import transforms
import cv2
from tqdm.auto import tqdm
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
import sys
sys.path.append('/home/sysadmin/Seongmin/SSV2A_Mul/Depth-Anything-V2')
from depth_anything_v2.dpt import DepthAnythingV2

DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'

# -----------------------------
# 1. Load Depth Model
# -----------------------------
def load_depth_model(device):
    model = DepthAnythingV2(encoder='vitl', features=256, out_channels=[256, 512, 1024, 1024])
    ckpt = '/home/sysadmin/Seongmin/SSV2A_Mul/Depth-Anything-V2/checkpoints/depth_anything_v2_vitl.pth?download=true'
    model.load_state_dict(torch.load(ckpt, map_location=device))
    return model.to(device).eval()


@torch.no_grad()
def precompute_depth(input_dir, output_dir, img_size=(224, 224), device="cpu", show_tqdm=True):
    """
    - input_dir 안에서 파일명에 '_depth'가 포함된 잘못 생성된 파일은 모두 삭제
    - RGB 이미지를 돌며 output_dir에 <stem>_depth.png 저장 (이미 있으면 skip)
    - rank 0에서만 실행하도록 main에서 제어
    """
    model = load_depth_model(device)
    input_dir = Path(input_dir)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # input_dir 안의 잘못된 *_depth* 파일 삭제
    wrongs = list(input_dir.rglob("*_depth*"))
    for p in (tqdm(wrongs, desc="Removing wrong *_depth files", leave=False) if show_tqdm else wrongs):
        try:
            p.unlink()
        except IsADirectoryError:
            # 혹시 디렉토리면 건너뛰기
            pass
        except FileNotFoundError:
            pass

    img_paths = list(input_dir.rglob("*.[jp][pn]g"))
    if show_tqdm:
        print(f"Found {len(img_paths)} images for depth precompute")

    iterator = tqdm(img_paths, desc="Precomputing Depth", leave=False) if show_tqdm else img_paths

    for img_path in iterator:
        rel_path = img_path.relative_to(input_dir)
        depth_path = (output_dir / rel_path).with_name(img_path.stem + "_depth.png")
        depth_path.parent.mkdir(parents=True, exist_ok=True)

        if depth_path.exists():
            continue

        img_cv = cv2.imread(str(img_path))
        if img_cv is None:
            continue
        img_rgb = img_cv[:, :, ::-1]  # BGR → RGB
        depth_map = model.infer_image(img_rgb)  # (H, W) float32
        depth_resized = cv2.resize(depth_map, img_size, interpolation=cv2.INTER_LINEAR)
        dmin, dmax = depth_resized.min(), depth_resized.max()
        depth_norm = (depth_resized - dmin) / (dmax - dmin + 1e-6)
        depth_img = (depth_norm * 255).astype('uint8')
        cv2.imwrite(str(depth_path), depth_img)


# -----------------------------
# 2. Dataset
# -----------------------------
class RGBDDataset(Dataset):
    def __init__(self, rgb_dir, depth_dir, transform=None, img_size=(224, 224)):
        self.rgb_dir = Path(rgb_dir)
        self.depth_dir = Path(depth_dir)
        self.transform = transform
        self.img_size = img_size

        self.samples = []
        self.class_to_idx = {}
        for idx, class_dir in enumerate(sorted(self.rgb_dir.iterdir())):
            if class_dir.is_dir():
                self.class_to_idx[class_dir.name] = idx
                for img_path in class_dir.glob("*.[jp][pn]g"):
                    depth_path = self.depth_dir / class_dir.name / f"{img_path.stem}_depth.png"
                    self.samples.append((img_path, depth_path, idx))

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        rgb_path, depth_path, label = self.samples[idx]

        rgb = cv2.imread(str(rgb_path))[:, :, ::-1]
        rgb = cv2.resize(rgb, self.img_size)
        depth = cv2.imread(str(depth_path), cv2.IMREAD_GRAYSCALE)
        depth = cv2.resize(depth, self.img_size)

        rgb = Image.fromarray(rgb)
        depth = Image.fromarray(depth)

        if self.transform:
            rgb = self.transform(rgb)
            depth = self.transform(depth)
        else:
            rgb = transforms.ToTensor()(rgb)
            depth = transforms.ToTensor()(depth)

        return rgb, depth, label


# -----------------------------
# 3. Model
# -----------------------------
class MultiScalePatchEmbed(nn.Module):
    def __init__(self, in_channels=3, embed_dims=[64, 128, 256, 512], patch_sizes=[4, 8, 16, 32]):
        super().__init__()
        self.stages = nn.ModuleList()
        for embed_dim, patch_size in zip(embed_dims, patch_sizes):
            self.stages.append(nn.Conv2d(in_channels, embed_dim, kernel_size=patch_size, stride=patch_size))

    def forward(self, x):
        multi_scale_feats = []
        for stage in self.stages:
            feat = stage(x)
            B, C, H, W = feat.shape
            feat = feat.flatten(2).transpose(1, 2)  # [B, N, C]
            multi_scale_feats.append(feat)
        return multi_scale_feats


class GeoAttention(nn.Module):
    def __init__(self, embed_dim, num_heads=4, dropout=0.1):
        super().__init__()
        self.mha = nn.MultiheadAttention(embed_dim=embed_dim, num_heads=num_heads, batch_first=True, dropout=dropout)
        self.norm = nn.LayerNorm(embed_dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, geom_prior=None):
        if geom_prior is not None:
            geom_prior = geom_prior / (geom_prior.max(dim=-1, keepdim=True)[0] + 1e-6)
        attn_out, _ = self.mha(x, x, x)
        if geom_prior is not None:
            geom_feat = torch.bmm(geom_prior, attn_out)
            x = x + geom_feat
        else:
            x = x + attn_out
        x = self.norm(x)
        return x


class ScaleTransformer(nn.Module):
    def __init__(self, embed_dim, num_layers=2, num_heads=4):
        super().__init__()
        self.layers = nn.ModuleList([GeoAttention(embed_dim, num_heads) for _ in range(num_layers)])

    def forward(self, x, geom_prior=None):
        for layer in self.layers:
            x = layer(x, geom_prior)
        return x


class RGBDepthDFormerClassifier(nn.Module):
    def __init__(self, num_classes, rgb_embed_dims=[64, 128, 256, 512], depth_embed_dims=[32, 64, 128, 256],
                 patch_sizes=[4, 8, 16, 32], num_heads=4, num_layers=2):
        super().__init__()
        self.rgb_embed = MultiScalePatchEmbed(in_channels=3, embed_dims=rgb_embed_dims, patch_sizes=patch_sizes)
        self.depth_embed = MultiScalePatchEmbed(in_channels=1, embed_dims=depth_embed_dims, patch_sizes=patch_sizes)

        self.transformers = nn.ModuleList()
        for rgb_dim, depth_dim in zip(rgb_embed_dims, depth_embed_dims):
            fused_dim = rgb_dim + depth_dim
            self.transformers.append(ScaleTransformer(fused_dim, num_layers=num_layers, num_heads=num_heads))

        self.classifier = nn.Sequential(
            nn.Linear(sum([r + d for r, d in zip(rgb_embed_dims, depth_embed_dims)]), 512),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(512, num_classes)
        )

    @staticmethod
    def compute_geometry_prior(depth_map, patch_size):
        B, _, H, W = depth_map.shape
        h_patches = H // patch_size
        w_patches = W // patch_size
        num_patches = h_patches * w_patches
        depth_patches = depth_map.unfold(2, patch_size, patch_size).unfold(3, patch_size, patch_size)
        depth_patches = depth_patches.contiguous().view(B, 1, num_patches, patch_size * patch_size)
        patch_mean = depth_patches.mean(dim=-1).squeeze(1)  # [B, N]
        geom_prior = torch.abs(patch_mean.unsqueeze(2) - patch_mean.unsqueeze(1))  # [B, N, N]
        return geom_prior

    def forward(self, rgb, depth):
        rgb_feats = self.rgb_embed(rgb)     # list of [B, N, D]
        depth_feats = self.depth_embed(depth)

        fused_feats = []
        for i, (rf, df) in enumerate(zip(rgb_feats, depth_feats)):
            x = torch.cat([rf, df], dim=-1)
            geom_prior = self.compute_geometry_prior(depth, patch_size=2 ** (i + 2))
            x = self.transformers[i](x, geom_prior)
            pooled = x.mean(dim=1)  # global pooling per scale
            fused_feats.append(pooled)

        multi_scale_feat = torch.cat(fused_feats, dim=-1)
        out = self.classifier(multi_scale_feat)
        return out

# -----------------------------
# 4. Train (DDP without AMP) with tqdm batch-level display
# -----------------------------
def train_ddp(train_dir, val_dir, epochs=20, batch_size=24, num_workers=8):
    import torch.distributed as dist
    from torch.nn.parallel import DistributedDataParallel as DDP
    from torch.utils.data import DataLoader, DistributedSampler
    from tqdm.auto import tqdm

    # DDP 초기화
    dist.init_process_group(backend="nccl")
    local_rank = int(os.environ["LOCAL_RANK"])
    torch.cuda.set_device(local_rank)
    device = torch.device("cuda", local_rank)
    rank = dist.get_rank()
    world_size = dist.get_world_size()

    # transform
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
    ])

    # 사전 depth 계산 (rank 0 전용)
    if rank == 0:
        precompute_depth("/4TB_2nd/Seongmin/scene/train", "/4TB_2nd/Seongmin/scene_depth/train",
                         img_size=(224, 224), device=device, show_tqdm=True)
        precompute_depth("/4TB_2nd/Seongmin/scene/val", "/4TB_2nd/Seongmin/scene_depth/val",
                         img_size=(224, 224), device=device, show_tqdm=True)
    dist.barrier()

    # Dataset & Sampler
    full_train_dataset = RGBDDataset(train_dir,
                                     "/4TB_2nd/Seongmin/scene_depth/train",
                                     transform=transform)
    train_dataset = RGBDDataset("/4TB_2nd/Seongmin/scene/train",
                                "/4TB_2nd/Seongmin/scene_depth/train",
                                transform=transform)
    val_dataset = RGBDDataset("/4TB_2nd/Seongmin/scene/val",
                              "/4TB_2nd/Seongmin/scene_depth/val",
                              transform=transform)

    train_sampler = DistributedSampler(train_dataset, shuffle=True, drop_last=False)
    val_sampler = DistributedSampler(val_dataset, shuffle=False, drop_last=False)

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        sampler=train_sampler,
        num_workers=num_workers,
        pin_memory=True,
        persistent_workers=(num_workers > 0),
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        sampler=val_sampler,
        num_workers=num_workers,
        pin_memory=True,
        persistent_workers=(num_workers > 0),
    )

    if rank == 0:
        print(f"[World {world_size}] Train samples: {len(train_dataset)} | Val samples: {len(val_dataset)}")

    num_classes = len(full_train_dataset.class_to_idx)

    # Model, Optim, Loss
    model = RGBDepthDFormerClassifier(num_classes=num_classes).to(device)
    model = DDP(model, device_ids=[local_rank], output_device=local_rank, find_unused_parameters=False)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-5)
    criterion = nn.CrossEntropyLoss()

    best_val_loss = float('inf')
    best_model_path = "best_model.pth"

    for epoch in range(epochs):
        model.train()
        train_sampler.set_epoch(epoch)

        if rank == 0:
            print(f"\nEpoch {epoch + 1}/{epochs}")

        total_loss = 0.0
        correct = 0
        seen = 0

        train_iter = tqdm(train_loader, desc="Training", leave=False) if rank == 0 else train_loader

        for rgb, depth, label in train_iter:
            rgb = rgb.to(device, non_blocking=True)
            depth = depth.to(device, non_blocking=True)
            label = label.to(device, non_blocking=True)

            optimizer.zero_grad(set_to_none=True)

            # ------------------------
            # AMP 비활성화 (float32) 적용
            # ------------------------
            logits = model(rgb, depth)
            loss = criterion(logits, label)

            loss.backward()
            optimizer.step()

            # 통계
            batch_sz = rgb.size(0)
            total_loss += loss.item() * batch_sz
            pred = logits.argmax(dim=1)
            correct += (pred == label).sum().item()
            seen += batch_sz

            # tqdm 실시간 표시
            if rank == 0:
                train_iter.set_postfix({"loss": f"{loss.item():.4f}", "acc": f"{correct/seen:.4f}"})

        # 모든 rank에서 합치기
        tensor_stats = torch.tensor([total_loss, correct, seen], dtype=torch.float64, device=device)
        dist.all_reduce(tensor_stats, op=dist.ReduceOp.SUM)
        total_loss_g, correct_g, seen_g = tensor_stats.tolist()
        train_loss = total_loss_g / max(1.0, seen_g)
        train_acc = correct_g / max(1.0, seen_g)

        if rank == 0:
            print(f"Train Loss: {train_loss:.4f} | Train Acc: {train_acc:.4f}")

        # -----------------
        # Validation
        # -----------------
        model.eval()
        val_loss = 0.0
        val_correct = 0
        val_seen = 0

        val_iter = tqdm(val_loader, desc="Validation", leave=False) if rank == 0 else val_loader

        with torch.no_grad():
            for rgb, depth, label in val_iter:
                rgb = rgb.to(device, non_blocking=True)
                depth = depth.to(device, non_blocking=True)
                label = label.to(device, non_blocking=True)

                logits = model(rgb, depth)
                loss = criterion(logits, label)

                batch_sz = rgb.size(0)
                val_loss += loss.item() * batch_sz
                pred = logits.argmax(dim=1)
                val_correct += (pred == label).sum().item()
                val_seen += batch_sz

                # tqdm 실시간 표시
                if rank == 0:
                    val_iter.set_postfix({"loss": f"{loss.item():.4f}", "acc": f"{val_correct/val_seen:.4f}"})

        tensor_stats = torch.tensor([val_loss, val_correct, val_seen], dtype=torch.float64, device=device)
        dist.all_reduce(tensor_stats, op=dist.ReduceOp.SUM)
        val_loss_g, val_correct_g, val_seen_g = tensor_stats.tolist()
        val_loss_mean = val_loss_g / max(1.0, val_seen_g)
        val_acc = val_correct_g / max(1.0, val_seen_g)

        if rank == 0:
            print(f"Val  Loss: {val_loss_mean:.4f} | Val  Acc: {val_acc:.4f}")

            if val_loss_mean < best_val_loss:
                best_val_loss = val_loss_mean
                torch.save(model.module.state_dict(), best_model_path)
                print(f"Saved best model at epoch {epoch + 1} | Val Loss: {val_loss_mean:.4f}")

    dist.barrier()
    dist.destroy_process_group()


# -----------------------------
# 5. 실행
# -----------------------------
if __name__ == "__main__":
    # 환경: torchrun이 LOCAL_RANK / RANK / WORLD_SIZE 세팅
    # 사용 경로는 네가 쓰던 경로와 동일하게 유지
    train_dir = "/4TB_2nd/Seongmin/scene/train"
    val_dir = "/4TB_2nd/Seongmin/scene/val"
    # DDP 학습 호출
    train_ddp(train_dir, val_dir, epochs=40, batch_size=8, num_workers=8)