"""
数据生成脚本：生成简单 3D 几何体的 2D 投影图像
"""

import numpy as np
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
    vertices = np.array([
        [1, 1, 1], [-1, -1, 1], [-1, 1, -1], [1, -1, -1]
    ], dtype=np.float32)
    
    faces = np.array([
        [0, 1, 2], [0, 1, 3], [0, 2, 3], [1, 2, 3]
    ], dtype=np.int32)
    
    return vertices, faces


def create_cylinder_mesh(n_segments=8):
    vertices = [[0, 0, -1], [0, 0, 1]]
    
    for i in range(n_segments):
        angle = 2 * np.pi * i / n_segments
        x, y = np.cos(angle), np.sin(angle)
        vertices.append([x, y, -1])
        vertices.append([x, y, 1])
    
    vertices = np.array(vertices, dtype=np.float32)
    faces = []
    
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
    
    for i in range(n_segments):
        angle = 2 * np.pi * i / n_segments
        x, y = np.cos(angle), np.sin(angle)
        vertices.append([x, y, -1])
    
    vertices = np.array(vertices, dtype=np.float32)
    faces = []
    
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
    
    x = rotated_vertices[:, 0]
    y = rotated_vertices[:, 1]
    z = rotated_vertices[:, 2]
    
    img = np.zeros((img_size, img_size), dtype=np.float32)
    z_buffer = np.full((img_size, img_size), -np.inf, dtype=np.float32)
    
    for face in faces:
        v0, v1, v2 = face
        
        pts_2d = np.array([
            [x[v0], y[v0]],
            [x[v1], y[v1]],
            [x[v2], y[v2]]
        ])
        
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
        'cube': create_cube_mesh(),
        'tetrahedron': create_tetrahedron_mesh(),
        'cylinder': create_cylinder_mesh(),
        'cone': create_cone_mesh()
    }
    
    dataset = []
    
    for obj_name, (vertices, faces) in meshes.items():
        print(f"生成 {obj_name} 数据...")
        
        for i in range(n_train):
            R = random_rotation_matrix()
            img = render_mesh(vertices, faces, R, img_size)
            
            img_path = output_dir / f"{obj_name}_train_{i:04d}.npy"
            np.save(img_path, img)
            
            dataset.append({
                'image': str(img_path),
                'rotation_matrix': R.tolist(),
                'object_type': obj_name,
                'split': 'train'
            })
        
        for i in range(n_test):
            R = random_rotation_matrix()
            img = render_mesh(vertices, faces, R, img_size)
            
            img_path = output_dir / f"{obj_name}_test_{i:04d}.npy"
            np.save(img_path, img)
            
            dataset.append({
                'image': str(img_path),
                'rotation_matrix': R.tolist(),
                'object_type': obj_name,
                'split': 'test'
            })
    
    metadata_path = output_dir / 'metadata.json'
    with open(metadata_path, 'w') as f:
        json.dump(dataset, f, indent=2)
    
    print(f"数据集生成完成！共 {len(dataset)} 个样本")
    return dataset


if __name__ == '__main__':
    import sys
    output_dir = sys.argv[1] if len(sys.argv) > 1 else 'data'
    generate_dataset(output_dir)
