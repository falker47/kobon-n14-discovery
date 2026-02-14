import json
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import itertools
from matplotlib.ticker import ScalarFormatter
import os

def generate_optical_aliasing_plot(json_path, output_pdf, output_png):
    # 1. Data Loading
    with open(json_path, 'r') as f:
        data = json.load(f)
    
    lines = data['lines']
    n = len(lines)
    thetas = np.array([l['theta'] for l in lines], dtype=np.float64)
    rhos = np.array([l['rho'] for l in lines], dtype=np.float64)
    
    print(f"Loaded {n} lines from {json_path}")
    
    # 2. Intersection Calculation (Cramer's Rule)
    # intersections[(i, j)] = (x, y)
    intersections = {}
    eps_parallel = 1e-12
    
    for i in range(n):
        for j in range(i + 1, n):
            det = np.sin(thetas[j] - thetas[i])
            if abs(det) < eps_parallel:
                continue
            
            x = (rhos[i] * np.sin(thetas[j]) - rhos[j] * np.sin(thetas[i])) / det
            y = (rhos[j] * np.cos(thetas[i]) - rhos[i] * np.cos(thetas[j])) / det
            intersections[(i, j)] = (x, y)

    # 3. Target Identification: Find smallest valid triangle
    valid_triangles = []
    eps_cut = 1e-9
    
    for i, j, k in itertools.combinations(range(n), 3):
        v1 = intersections.get((i, j))
        v2 = intersections.get((j, k))
        v3 = intersections.get((i, k))
        
        if v1 is None or v2 is None or v3 is None:
            continue
            
        verts = [v1, v2, v3]
        is_valid = True
        
        # Segment intersection test (against all other lines)
        for m in range(n):
            if m in (i, j, k):
                continue
            
            dists = []
            for vx, vy in verts:
                d = vx * np.cos(thetas[m]) + vy * np.sin(thetas[m]) - rhos[m]
                dists.append(d)
            
            # If line m separates vertices, it cuts the triangle
            d_min, d_max = min(dists), max(dists)
            if d_min < -eps_cut and d_max > eps_cut:
                is_valid = False
                break
        
        if is_valid:
            # Area using Shoelace formula
            area = 0.5 * abs(v1[0]*(v2[1] - v3[1]) + v2[0]*(v3[1] - v1[1]) + v3[0]*(v1[1] - v2[1]))
            valid_triangles.append({
                'lines': (i, j, k),
                'verts': verts,
                'area': area
            })

    print(f"Found {len(valid_triangles)} valid triangles.")
    
    if not valid_triangles:
        print("No valid triangles found!")
        return

    # Find the triangle with the smallest strictly positive area
    smallest_tri = min(valid_triangles, key=lambda t: t['area'] if t['area'] > 1e-18 else float('inf'))
    
    print(f"Smallest triangle area: {smallest_tri['area']:.2e}")
    print(f"Lines: {smallest_tri['lines']}")
    
    # 4. Micro-Rendering
    fig, ax = plt.subplots(figsize=(10, 10))
    
    verts = np.array(smallest_tri['verts'])
    x_coords, y_coords = verts[:, 0], verts[:, 1]
    
    # Calculate centroid and max edge length for dynamic padding
    cx, cy = x_coords.mean(), y_coords.mean()
    
    # Distances between vertices (Shoelace vertices: v1, v2, v3)
    v1, v2, v3 = verts[0], verts[1], verts[2]
    edges = [
        np.linalg.norm(v1 - v2),
        np.linalg.norm(v2 - v3),
        np.linalg.norm(v3 - v1)
    ]
    max_edge = max(edges)
    
    # Dynamic Padding: Centroid +/- (1.5 * max_edge)
    span = 1.5 * max_edge
    ax.set_xlim(cx - span, cx + span)
    ax.set_ylim(cy - span, cy + span)
    
    # Area Fill: Polygon highlighting the gap
    from matplotlib.patches import Polygon
    poly = Polygon(verts, facecolor='red', alpha=0.3, edgecolor='none', zorder=2)
    ax.add_patch(poly)
    
    # Vertex Highlighting
    ax.scatter(x_coords, y_coords, color='red', s=15, zorder=5)
    
    # Plot the 3 lines forming the triangle with ax.axline for extreme precision stability
    # Line 1 passes through v1 and v2
    ax.axline(xy1=v1, xy2=v2, color='black', linewidth=0.2, zorder=1)
    # Line 2 passes through v2 and v3
    ax.axline(xy1=v2, xy2=v3, color='black', linewidth=0.2, zorder=1)
    # Line 3 passes through v3 and v1
    ax.axline(xy1=v3, xy2=v1, color='black', linewidth=0.2, zorder=1)

    # Style axes
    ax.set_aspect('equal')
    formatter = ScalarFormatter(useOffset=False, useMathText=True)
    formatter.set_scientific(True)
    formatter.set_powerlimits((-2, 2))
    ax.xaxis.set_major_formatter(formatter)
    ax.yaxis.set_major_formatter(formatter)
    
    # Remove titles as requested
    plt.tight_layout()
    
    # Create assets directory if it doesn't exist
    os.makedirs(os.path.dirname(output_pdf), exist_ok=True)
    
    # Export at 1200 DPI
    plt.savefig(output_pdf, format='pdf', dpi=1200)
    plt.savefig(output_png, format='png', dpi=1200)
    print(f"Saved refined plots to {output_pdf} and {output_png}")

if __name__ == "__main__":
    json_path = "data/solutions_run2/N14_K54_3c335303.json"
    output_pdf = "assets/optical_aliasing_zoom.pdf"
    output_png = "assets/optical_aliasing_zoom.png"
    
    # Fallback to alternative path if not found
    if not os.path.exists(json_path):
        json_path = "data/N14_K54_3c335303.json"
        
    generate_optical_aliasing_plot(json_path, output_pdf, output_png)
