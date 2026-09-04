"""
训练脚本：两组实验
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import numpy as np
import json
from pathlib import Path
import matplotlib.pyplot as plt

from model import SimpleRotationCNN, rotation_6d_to_matrix, compute_geodesic_distance
from differentiable_render import DifferentiableRenderer, generate_gt_silhouette
from generate_data import create_cube_mesh, create_tetrahedron_mesh, create_cylinder_mesh, create_cone_mesh


class RotationDataset(Dataset):
    def __init__(self, metadata_path, split='train'):
        with open(metadata_path, 'r') as f:
            metadata = json.load(f)
        self.samples = [s for s in metadata if s['split'] == split]
    
    def __len__(self):
        return len(self.samples)
    
    def __getitem__(self, idx):
        sample = self.samples[idx]
        img = np.load(sample['image'])
        img = torch.tensor(img, dtype=torch.float32).unsqueeze(0)
        R = torch.tensor(sample['rotation_matrix'], dtype=torch.float32)
        obj_type = sample['object_type']
        return img, R, obj_type


def collate_fn(batch):
    images, rotations, obj_types = zip(*batch)
    return torch.stack(images), torch.stack(rotations), list(obj_types)


def train_one_epoch(model, dataloader, optimizer, criterion, device, renderer_dict=None, lambda_render=0.1):
    model.train()
    total_loss = 0
    total_angle_error = 0
    n_batches = 0
    
    for images, R_gt, obj_types in dataloader:
        images = images.to(device)
        R_gt = R_gt.to(device)
        batch_size = images.shape[0]
        
        rotation_6d = model(images)
        R_pred = rotation_6d_to_matrix(rotation_6d)
        
        loss_rotation = nn.functional.mse_loss(R_pred, R_gt)
        
        if renderer_dict is not None:
            loss_render = 0
            for i, obj_type in enumerate(obj_types):
                renderer = renderer_dict[obj_type]
                pred_silhouette = renderer(R_pred[i:i+1])
                gt_silhouette = generate_gt_silhouette(
                    renderer.vertices.cpu().numpy(),
                    renderer.faces.cpu().numpy(),
                    R_gt[i:i+1],
                    img_size=64
                ).to(device)
                loss_render += nn.functional.mse_loss(pred_silhouette, gt_silhouette)
            loss_render /= batch_size
            loss = loss_rotation + lambda_render * loss_render
        else:
            loss = loss_rotation
        
        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()
        
        with torch.no_grad():
            angles = compute_geodesic_distance(R_pred, R_gt)
            total_angle_error += angles.mean().item()
        
        total_loss += loss.item()
        n_batches += 1
    
    return total_loss / n_batches, total_angle_error / n_batches


def evaluate(model, dataloader, device):
    model.eval()
    total_angle_error = 0
    n_batches = 0
    
    with torch.no_grad():
        for images, R_gt, obj_types in dataloader:
            images = images.to(device)
            R_gt = R_gt.to(device)
            rotation_6d = model(images)
            R_pred = rotation_6d_to_matrix(rotation_6d)
            angles = compute_geodesic_distance(R_pred, R_gt)
            total_angle_error += angles.mean().item()
            n_batches += 1
    
    return total_angle_error / n_batches


def plot_training_curves(train_losses, val_angle_errors, save_path):
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))
    
    ax1.plot(train_losses)
    ax1.set_xlabel('Epoch')
    ax1.set_ylabel('Training Loss')
    ax1.set_title('Training Loss')
    ax1.grid(True)
    
    ax2.plot(val_angle_errors)
    ax2.set_xlabel('Epoch')
    ax2.set_ylabel('Mean Angle Error (deg)')
    ax2.set_title('Validation Angle Error')
    ax2.grid(True)
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    print(f"训练曲线已保存到: {save_path}")


def main(experiment_name, use_render=False, lambda_render=0.1, n_epochs=30, batch_size=16, lr=1e-3):
    print(f"\n{'='*60}")
    print(f"实验: {experiment_name}")
    print(f"使用可微分渲染: {use_render}")
    print(f"{'='*60}\n")
    
    device = torch.device('cpu')
    print(f"使用设备: {device}")
    
    data_dir = Path('data')
    metadata_path = data_dir / 'metadata.json'
    
    train_dataset = RotationDataset(metadata_path, split='train')
    test_dataset = RotationDataset(metadata_path, split='test')
    
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, collate_fn=collate_fn)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, collate_fn=collate_fn)
    
    print(f"训练集大小: {len(train_dataset)}")
    print(f"测试集大小: {len(test_dataset)}")
    
    model = SimpleRotationCNN().to(device)
    print(f"模型参数量: {sum(p.numel() for p in model.parameters()):,}")
    
    optimizer = optim.Adam(model.parameters(), lr=lr)
    criterion = nn.MSELoss()
    
    renderer_dict = None
    if use_render:
        print("初始化可微分渲染器...")
        meshes = {
            'cube': create_cube_mesh(),
            'tetrahedron': create_tetrahedron_mesh(),
            'cylinder': create_cylinder_mesh(),
            'cone': create_cone_mesh()
        }
        renderer_dict = {
            obj_type: DifferentiableRenderer(vertices, faces, img_size=64).to(device)
            for obj_type, (vertices, faces) in meshes.items()
        }
        print(f"已初始化 {len(renderer_dict)} 个渲染器")
    
    train_losses = []
    val_angle_errors = []
    
    print(f"\n开始训练 {n_epochs} 个 epoch...\n")
    
    for epoch in range(n_epochs):
        train_loss, train_angle_error = train_one_epoch(
            model, train_loader, optimizer, criterion, device,
            renderer_dict=renderer_dict, lambda_render=lambda_render
        )
        
        val_angle_error = evaluate(model, test_loader, device)
        
        train_losses.append(train_loss)
        val_angle_errors.append(val_angle_error)
        
        print(f"Epoch {epoch+1:02d}/{n_epochs} | "
              f"Train Loss: {train_loss:.4f} | "
              f"Train MAE: {train_angle_error:.2f} deg | "
              f"Val MAE: {val_angle_error:.2f} deg")
    
    results_dir = Path('results')
    results_dir.mkdir(exist_ok=True)
    
    plot_training_curves(train_losses, val_angle_errors, results_dir / f'{experiment_name}_curves.png')
    
    model_path = results_dir / f'{experiment_name}_model.pth'
    torch.save(model.state_dict(), model_path)
    print(f"模型已保存到: {model_path}")
    
    history = {
        'experiment': experiment_name,
        'use_render': use_render,
        'lambda_render': lambda_render,
        'n_epochs': n_epochs,
        'train_losses': train_losses,
        'val_angle_errors': val_angle_errors,
        'final_val_mae': val_angle_errors[-1]
    }
    
    history_path = results_dir / f'{experiment_name}_history.json'
    with open(history_path, 'w') as f:
        json.dump(history, f, indent=2)
    
    print(f"\n最终验证 MAE: {val_angle_errors[-1]:.2f} deg")
    print(f"训练历史已保存到: {history_path}")
    
    return history


if __name__ == '__main__':
    print("开始实验...")
    
    history_baseline = main('baseline', use_render=False, n_epochs=30)
    history_render = main('with_render', use_render=True, lambda_render=0.1, n_epochs=30)
    
    print("\n" + "="*60)
    print("实验完成！")
    print("="*60)
    print(f"Baseline 最终 MAE: {history_baseline['final_val_mae']:.2f} deg")
    print(f"+Render 最终 MAE: {history_render['final_val_mae']:.2f} deg")
