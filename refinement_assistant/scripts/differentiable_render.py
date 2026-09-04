"""
可微分渲染器：将 3D mesh 投影到 2D，生成 soft silhouette
"""

import torch
import torch.nn as nn
import numpy as np


class DifferentiableRenderer(nn.Module):
    def __init__(self, vertices, faces, img_size=64, sigma=0.1):
        super().__init__()
        
        self.img_size = img_size
        self.sigma = sigma
        
        self.register_buffer('vertices', torch.tensor(vertices, dtype=torch.float32))
        self.register_buffer('faces', torch.tensor(faces, dtype=torch.long))
        
        y, x = torch.meshgrid(
            torch.linspace(-1, 1, img_size),
            torch.linspace(-1, 1, img_size),
            indexing='ij'
        )
        self.register_buffer('pixel_grid', torch.stack([x, y], dim=-1))
    
    def forward(self, rotation_matrix):
        batch_size = rotation_matrix.shape[0]
        rotated_vertices = torch.einsum('bij,vj->bvi', rotation_matrix, self.vertices)
        projected_2d = rotated_vertices[:, :, :2]
        silhouette = self._render_soft_silhouette(projected_2d)
        return silhouette
    
    def _render_soft_silhouette(self, projected_2d):
        batch_size = projected_2d.shape[0]
        pixels = self.pixel_grid.view(-1, 2)
        
        projected_2d_expanded = projected_2d.unsqueeze(1)
        pixels_expanded = pixels.unsqueeze(0).unsqueeze(2).to(projected_2d.device)
        
        distances = torch.sum((projected_2d_expanded - pixels_expanded) ** 2, dim=-1)
        soft_min_dist = -self.sigma * torch.logsumexp(-distances / self.sigma, dim=-1)
        silhouette = torch.exp(-soft_min_dist / self.sigma)
        silhouette = silhouette.view(batch_size, self.img_size, self.img_size)
        
        return silhouette


def generate_gt_silhouette(vertices, faces, rotation_matrix, img_size=64):
    batch_size = rotation_matrix.shape[0]
    silhouettes = []
    
    for b in range(batch_size):
        R = rotation_matrix[b].cpu().numpy()
        rotated_vertices = vertices @ R.T
        
        img = np.zeros((img_size, img_size), dtype=np.float32)
        z_buffer = np.full((img_size, img_size), -np.inf, dtype=np.float32)
        
        for face in faces:
            v0, v1, v2 = face
            
            pts_2d = np.array([
                [rotated_vertices[v0, 0], rotated_vertices[v0, 1]],
                [rotated_vertices[v1, 0], rotated_vertices[v1, 1]],
                [rotated_vertices[v2, 0], rotated_vertices[v2, 1]]
            ])
            
            pts_img = ((pts_2d + 1) / 2 * img_size).astype(int)
            pts_img = np.clip(pts_img, 0, img_size - 1)
            
            avg_z = (rotated_vertices[v0, 2] + rotated_vertices[v1, 2] + rotated_vertices[v2, 2]) / 3
            
            min_y, max_y = pts_img[:, 1].min(), pts_img[:, 1].max()
            min_x, max_x = pts_img[:, 0].min(), pts_img[:, 0].max()
            
            for py in range(min_y, max_y + 1):
                for px in range(min_x, max_x + 1):
                    if point_in_triangle(px, py, pts_img[0], pts_img[1], pts_img[2]):
                        if avg_z > z_buffer[py, px]:
                            z_buffer[py, px] = avg_z
                            img[py, px] = 1.0
        
        silhouettes.append(img)
    
    return torch.tensor(np.array(silhouettes), dtype=torch.float32)


def point_in_triangle(px, py, v0, v1, v2):
    def sign(p1, p2, p3):
        return (p1[0] - p3[0]) * (p2[1] - p3[1]) - (p2[0] - p3[0]) * (p1[1] - p3[1])
    
    d1 = sign((px, py), v0, v1)
    d2 = sign((px, py), v1, v2)
    d3 = sign((px, py), v2, v0)
    
    has_neg = (d1 < 0) or (d2 < 0) or (d3 < 0)
    has_pos = (d1 > 0) or (d2 > 0) or (d3 > 0)
    
    return not (has_neg and has_pos)
