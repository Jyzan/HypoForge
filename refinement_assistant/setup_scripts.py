import os
os.makedirs('scripts', exist_ok=True)

# 创建 generate_data.py
with open('scripts/generate_data.py', 'w', encoding='utf-8') as f:
    f.write('''import numpy as np
import torch
from pathlib import Path
import json

def create_cube_mesh():
    vertices = np.array([
        [-1, -1, -1], [1, -1, -1], [1, 1, -1], [-1, 1, -1],
        [-1, -1, 1], [1, -1, 1], [1, 1, 1], [-1, 1, 1]
    ], dtype=np.float32)
    faces = np.array([
        [0, 1, 2], [0, 2, 3], [4, 5, 6], [4, 6, 7],
        [0, 1, 5], [0, 5, 4], [2, 3, 7], [2, 7, 6],
        [0, 3, 7], [0, 7, 4], [1, 2, 6], [1, 6, 5],
    ], dtype=np.int32)
    return vertices, faces

def create_tetrahedron_mesh():
    vertices = np.array([[1, 1, 1], [-1, -1, 1], [-1, 1, -1], [1, -1, -1]], dtype=np.float32)
    faces = np.array([[0, 1, 2], [0, 1, 3], [0, 2, 3], [1, 2, 3]], dtype=np.int32)
    return vertices, faces

def create_cylinder_mesh(n_segments=8):
    vertices = [[0, 0, -1], [0, 0, 1]]
    faces = []
    for i in range(n_segments):
        angle = 2 * np.pi * i / n_segments
        x, y = np.cos(angle), np.sin(angle)
        vertices.append([x, y, -1])
        vertices.append([x, y, 1])
    vertices = np.array(vertices, dtype=np.float32)
    for i in range(n_segments):
        next_i = (i + 1) % n_segments
        faces.append([0, 2 + 2*i, 2 + 2*next_i])
    for i in range(n_segments):
        next_i = (i + 1) % n_segments
        faces.append([1, 3 + 2*next_i, 3 + 2*i])
    for i in range(n_segments):
        next_i = (i + 1) % n_segments
        faces.append([2 + 2*i, 2 + 2*next_i, 3 + 2*next_i])
        faces.append([2 + 2*i, 3 + 2*next_i, 3 + 2*i])
    faces = np.array(faces, dtype=np.int32)
    return vertices, faces

def create_cone_mesh(n_segments=8):
    vertices = [[0, 0, -1], [0, 0, 1]]
    faces = []
    for i in range(n_segments):
        angle = 2 * np.pi * i / n_segments
        x, y = np.cos(angle), np.sin(angle)
        vertices.append([x, y, -1])
    vertices = np.array(vertices, dtype=np.float32)
    for i in range(n_segments):
        next_i = (i + 1) % n_segments
        faces.append([0, 2 + i, 2 + next_i])
    for i in range(n_segments):
        next_i = (i + 1) % n_segments
        faces.append([1, 2 + next_i, 2 + i])
    faces = np.array(faces, dtype=np.int32)
    return vertices, faces

def random_rotation_matrix():
    u1, u2, u3 = np.random.random(3)
    q = np.array([
        np.sqrt(1 - u1) * np.sin(2 * np.pi * u2),
        np.sqrt(1 - u1) * np.cos(2 * np.pi * u2),
        np.sqrt(u1) * np.sin(2 * np.pi * u3),
        np.sqrt(u1) * np.cos(2 * np.pi * u3)
    ])
    R = np.array([
        [1 - 2*q[1]**2 - 2*q[2]**2, 2*q[0]*q[1] - 2*q[2]*q[3], 2*q[0]*q[2] + 2*q[1]*q[3]],
        [2*q[0]*q[1] + 2*q[2]*q[3], 1 - 2*q[0]**2 - 2*q[2]**2, 2*q[1]*q[2] - 2*q[0]*q[3]],
        [2*q[0]*q[2] - 2*q[1]*q[3], 2*q[1]*q[2] + 2*q[0]*q[3], 1 - 2*q[0]**2 - 2*q[1]**2]
    ])
    return R.astype(np.float32)

def point_in_triangle(px, py, v0, v1, v2):
    def sign(p1, p2, p3):
        return (p1[0] - p3[0]) * (p2[1] - p3[1]) - (p2[0] - p3[0]) * (p1[1] - p3[1])
    d1 = sign((px, py), v0, v1)
    d2 = sign((px, py), v1, v2)
    d3 = sign((px, py), v2, v0)
    has_neg = (d1 < 0) or (d2 < 0) or (d3 < 0)
    has_pos = (d1 > 0) or (d2 > 0) or (d3 > 0)
    return not (has_neg and has_pos)

def render_mesh(vertices, faces, rotation_matrix, img_size=64):
    rotated_vertices = vertices @ rotation_matrix.T
    x, y, z = rotated_vertices[:, 0], rotated_vertices[:, 1], rotated_vertices[:, 2]
    img = np.zeros((img_size, img_size), dtype=np.float32)
    z_buffer = np.full((img_size, img_size), -np.inf, dtype=np.float32)
    for face in faces:
        v0, v1, v2 = face
        pts_2d = np.array([[x[v0], y[v0]], [x[v1], y[v1]], [x[v2], y[v2]]])
        pts_img = ((pts_2d + 1) / 2 * img_size).astype(int)
        pts_img = np.clip(pts_img, 0, img_size - 1)
        avg_z = (z[v0] + z[v1] + z[v2]) / 3
        min_y, max_y = pts_img[:, 1].min(), pts_img[:, 1].max()
        min_x, max_x = pts_img[:, 0].min(), pts_img[:, 0].max()
        for py in range(min_y, max_y + 1):
            for px in range(min_x, max_x + 1):
                if point_in_triangle(px, py, pts_img[0], pts_img[1], pts_img[2]):
                    if avg_z > z_buffer[py, px]:
                        z_buffer[py, px] = avg_z
                        img[py, px] = 1.0
    return img

def generate_dataset(output_dir, n_train=100, n_test=25, img_size=64):
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    meshes = {
        "cube": create_cube_mesh(),
        "tetrahedron": create_tetrahedron_mesh(),
        "cylinder": create_cylinder_mesh(),
        "cone": create_cone_mesh()
    }
    dataset = []
    for obj_name, (vertices, faces) in meshes.items():
        print(f"生成 {obj_name} 数据...")
        for i in range(n_train):
            R = random_rotation_matrix()
            img = render_mesh(vertices, faces, R, img_size)
            img_path = output_dir / f"{obj_name}_train_{i:04d}.npy"
            np.save(img_path, img)
            dataset.append({"image": str(img_path), "rotation_matrix": R.tolist(), "object_type": obj_name, "split": "train"})
        for i in range(n_test):
            R = random_rotation_matrix()
            img = render_mesh(vertices, faces, R, img_size)
            img_path = output_dir / f"{obj_name}_test_{i:04d}.npy"
            np.save(img_path, img)
            dataset.append({"image": str(img_path), "rotation_matrix": R.tolist(), "object_type": obj_name, "split": "test"})
    metadata_path = output_dir / "metadata.json"
    with open(metadata_path, "w") as f:
        json.dump(dataset, f, indent=2)
    print(f"数据集生成完成！共 {len(dataset)} 个样本")
    return dataset

if __name__ == "__main__":
    import sys
    output_dir = sys.argv[1] if len(sys.argv) > 1 else "data"
    generate_dataset(output_dir)
''')

print('✅ generate_data.py 已创建')

# 创建 model.py
with open('scripts/model.py', 'w', encoding='utf-8') as f:
    f.write('''import torch
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

if __name__ == "__main__":
    model = SimpleRotationCNN()
    print(f"模型参数量: {sum(p.numel() for p in model.parameters()):,}")
    x = torch.randn(4, 1, 64, 64)
    rotation_6d = model(x)
    print(f"输入形状: {x.shape}")
    print(f"6D 旋转输出形状: {rotation_6d.shape}")
    R = rotation_6d_to_matrix(rotation_6d)
    print(f"旋转矩阵形状: {R.shape}")
''')

print('✅ model.py 已创建')
