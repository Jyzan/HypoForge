"""
极简 CNN 模型：模拟 YOLO 检测头的旋转回归分支
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class SimpleRotationCNN(nn.Module):
    def __init__(self):
        super().__init__()
        
        self.features = nn.Sequential(
            nn.Conv2d(1, 16, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(16, 32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.AdaptiveAvgPool2d(1)
        )
        
        self.regressor = nn.Sequential(
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, 6)
        )
    
    def forward(self, x):
        features = self.features(x)
        features = features.view(features.size(0), -1)
        rotation_6d = self.regressor(features)
        return rotation_6d


def rotation_6d_to_matrix(rotation_6d):
    a1 = rotation_6d[:, :3]
    a2 = rotation_6d[:, 3:]
    
    b1 = F.normalize(a1, p=2, dim=1)
    dot_product = (a2 * b1).sum(dim=1, keepdim=True)
    b2 = a2 - dot_product * b1
    b2 = F.normalize(b2, p=2, dim=1)
    b3 = torch.cross(b1, b2, dim=1)
    
    rotation_matrix = torch.stack([b1, b2, b3], dim=2)
    return rotation_matrix


def compute_geodesic_distance(R_pred, R_gt):
    R_diff = torch.bmm(R_pred.transpose(1, 2), R_gt)
    trace = R_diff[:, 0, 0] + R_diff[:, 1, 1] + R_diff[:, 2, 2]
    trace = torch.clamp(trace, -1.0, 3.0)
    cos_theta = (trace - 1.0) / 2.0
    cos_theta = torch.clamp(cos_theta, -1.0, 1.0)
    angles = torch.acos(cos_theta)
    angles_deg = angles * 180.0 / torch.pi
    return angles_deg


if __name__ == '__main__':
    model = SimpleRotationCNN()
    print(f"模型参数量: {sum(p.numel() for p in model.parameters()):,}")
    
    x = torch.randn(4, 1, 64, 64)
    rotation_6d = model(x)
    print(f"输入形状: {x.shape}")
    print(f"6D 旋转输出形状: {rotation_6d.shape}")
    
    R = rotation_6d_to_matrix(rotation_6d)
    print(f"旋转矩阵形状: {R.shape}")
