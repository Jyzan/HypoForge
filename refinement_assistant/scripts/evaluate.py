"""
评估脚本：可视化预测结果
"""

import torch
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import json

from model import SimpleRotationCNN, rotation_6d_to_matrix, compute_geodesic_distance
from differentiable_render import DifferentiableRenderer, generate_gt_silhouette
from generate_data import create_cube_mesh, create_tetrahedron_mesh, create_cylinder_mesh, create_cone_mesh
from train import RotationDataset, collate_fn
from torch.utils.data import DataLoader


def visualize_predictions(model, dataloader, device, renderer_dict, n_samples=8):
    model.eval()
    
    images, R_gt, obj_types = next(iter(dataloader))
    images = images.to(device)
    R_gt = R_gt.to(device)
    
    with torch.no_grad():
        rotation_6d = model(images)
        R_pred = rotation_6d_to_matrix(rotation_6d)
    
    angles = compute_geodesic_distance(R_pred, R_gt)
    
    n_samples = min(n_samples, len(images))
    fig, axes = plt.subplots(n_samples, 3, figsize=(12, 4 * n_samples))
    
    if n_samples == 1:
        axes = axes.reshape(1, -1)
    
    for i in range(n_samples):
        obj_type = obj_types[i]
        renderer = renderer_dict[obj_type]
        
        axes[i, 0].imshow(images[i, 0].cpu().numpy(), cmap='gray')
        axes[i, 0].set_title(f'Input ({obj_type})\nMAE: {angles[i]:.2f} deg')
        axes[i, 0].axis('off')
        
        pred_silhouette = renderer(R_pred[i:i+1])[0].cpu().numpy()
        axes[i, 1].imshow(pred_silhouette, cmap='gray')
        axes[i, 1].set_title('Predicted Silhouette')
        axes[i, 1].axis('off')
        
        gt_silhouette = generate_gt_silhouette(
            renderer.vertices.cpu().numpy(),
            renderer.faces.cpu().numpy(),
            R_gt[i:i+1],
            img_size=64
        )[0].numpy()
        axes[i, 2].imshow(gt_silhouette, cmap='gray')
        axes[i, 2].set_title('Ground Truth Silhouette')
        axes[i, 2].axis('off')
    
    plt.tight_layout()
    plt.savefig('results/predictions.png', dpi=150, bbox_inches='tight')
    print("可视化结果已保存到: results/predictions.png")


def compare_experiments():
    results_dir = Path('results')
    
    with open(results_dir / 'baseline_history.json', 'r') as f:
        baseline = json.load(f)
    
    with open(results_dir / 'with_render_history.json', 'r') as f:
        with_render = json.load(f)
    
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    
    axes[0].plot(baseline['train_losses'], label='Baseline', linewidth=2)
    axes[0].plot(with_render['train_losses'], label='+Render', linewidth=2)
    axes[0].set_xlabel('Epoch', fontsize=12)
    axes[0].set_ylabel('Training Loss', fontsize=12)
    axes[0].set_title('Training Loss Comparison', fontsize=14)
    axes[0].legend(fontsize=11)
    axes[0].grid(True, alpha=0.3)
    
    axes[1].plot(baseline['val_angle_errors'], label='Baseline', linewidth=2)
    axes[1].plot(with_render['val_angle_errors'], label='+Render', linewidth=2)
    axes[1].set_xlabel('Epoch', fontsize=12)
    axes[1].set_ylabel('Mean Angle Error (deg)', fontsize=12)
    axes[1].set_title('Validation MAE Comparison', fontsize=14)
    axes[1].legend(fontsize=11)
    axes[1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(results_dir / 'comparison.png', dpi=150, bbox_inches='tight')
    print("对比图已保存到: results/comparison.png")
    
    print("\n" + "="*60)
    print("实验结果总结")
    print("="*60)
    print(f"Baseline:")
    print(f"  最终验证 MAE: {baseline['final_val_mae']:.2f} deg")
    print(f"  最终训练 Loss: {baseline['train_losses'][-1]:.4f}")
    print(f"\n+Render:")
    print(f"  最终验证 MAE: {with_render['final_val_mae']:.2f} deg")
    print(f"  最终训练 Loss: {with_render['train_losses'][-1]:.4f}")
    print(f"\n改进:")
    improvement = baseline['final_val_mae'] - with_render['final_val_mae']
    print(f"  MAE 降低: {improvement:.2f} deg ({improvement/baseline['final_val_mae']*100:.1f}%)")


def main():
    print("开始评估...")
    
    device = torch.device('cpu')
    
    data_dir = Path('data')
    metadata_path = data_dir / 'metadata.json'
    test_dataset = RotationDataset(metadata_path, split='test')
    test_loader = DataLoader(test_dataset, batch_size=8, shuffle=False, collate_fn=collate_fn)
    
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
    
    print("\n评估 Baseline 模型...")
    model_baseline = SimpleRotationCNN().to(device)
    model_baseline.load_state_dict(torch.load('results/baseline_model.pth', map_location=device))
    visualize_predictions(model_baseline, test_loader, device, renderer_dict, n_samples=4)
    
    print("\n评估 +Render 模型...")
    model_render = SimpleRotationCNN().to(device)
    model_render.load_state_dict(torch.load('results/with_render_model.pth', map_location=device))
    
    compare_experiments()
    
    print("\n评估完成！")


if __name__ == '__main__':
    main()
