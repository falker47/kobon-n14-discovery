"""
VERIFY_GRAIL_OEIS.PY — OEIS-Compliant Minimalist Renderer
==========================================================
Standalone script. Pure topological illustration for OEIS submission.
Produces: assets/Kobon_N14_K54_OEIS.png
Requirements: Black lines on white background, no axes, no fills, no text.
"""
import os
import json
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from itertools import combinations

# ─── CONFIGURATION ──────────────────────────────────────────────────────────
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
TARGET_FILE  = os.path.join(PROJECT_ROOT, "data", "solutions_run2", "N14_K54_3c335303.json")
OUTPUT_IMAGE = os.path.join(PROJECT_ROOT, "assets", "Kobon_N14_K54_OEIS.png")

EPSILON      = 1e-9
DPI          = 1200
LINE_WIDTH   = 0.6
# ────────────────────────────────────────────────────────────────────────────


def load_lines(filepath):
    if not os.path.isfile(filepath):
        raise FileNotFoundError(f"CRITICAL: {filepath} not found.")
    with open(filepath, "r") as f:
        data = json.load(f)
    lines = sorted(data["lines"], key=lambda x: x["id"])
    return np.array([[l["theta"], l["rho"]] for l in lines], dtype=np.float64)


def solve_intersections(genome):
    n = len(genome)
    intersections = {}
    for i in range(n):
        for j in range(i + 1, n):
            th_i, rho_i = genome[i]
            th_j, rho_j = genome[j]
            det = np.sin(th_j - th_i)
            if abs(det) < EPSILON:
                continue
            x = (rho_i * np.sin(th_j) - rho_j * np.sin(th_i)) / det
            y = (rho_j * np.cos(th_i) - rho_i * np.cos(th_j)) / det
            intersections[(i, j)] = (x, y)
    return intersections


def is_valid_triangle(genome, triplet, intersections):
    i, j, k = triplet
    p1 = intersections.get((min(i, j), max(i, j)))
    p2 = intersections.get((min(j, k), max(j, k)))
    p3 = intersections.get((min(i, k), max(i, k)))
    if p1 is None or p2 is None or p3 is None:
        return None
    n = len(genome)
    for m in range(n):
        if m in triplet:
            continue
        th_m, rho_m = genome[m]
        cos_m, sin_m = np.cos(th_m), np.sin(th_m)
        d1 = p1[0] * cos_m + p1[1] * sin_m - rho_m
        d2 = p2[0] * cos_m + p2[1] * sin_m - rho_m
        d3 = p3[0] * cos_m + p3[1] * sin_m - rho_m
        if (d1 * d2) < -EPSILON or (d2 * d3) < -EPSILON or (d3 * d1) < -EPSILON:
            return None
    return [p1, p2, p3]


def find_valid_triangles(genome):
    n = len(genome)
    intersections = solve_intersections(genome)
    valid = []
    for t in combinations(range(n), 3):
        verts = is_valid_triangle(genome, t, intersections)
        if verts is not None:
            valid.append((t, verts))
    return valid


def render_oeis(genome, valid_triangles):
    """Pure minimalist rendering for OEIS compliance."""
    # Compute dynamic bounding box from triangle vertices
    all_verts = []
    for _, verts in valid_triangles:
        all_verts.extend(verts)
    all_verts = np.array(all_verts)
    
    x_min, x_max = all_verts[:, 0].min(), all_verts[:, 0].max()
    y_min, y_max = all_verts[:, 1].min(), all_verts[:, 1].max()
    pad_x = (x_max - x_min) * 0.05
    pad_y = (y_max - y_min) * 0.05
    x_lo, x_hi = x_min - pad_x, x_max + pad_x
    y_lo, y_hi = y_min - pad_y, y_max + pad_y

    fig, ax = plt.subplots(figsize=(30, 30))
    ax.set_facecolor("white")
    fig.patch.set_facecolor("white")

    # Draw lines (strictly solid black)
    x_range = np.linspace(x_lo, x_hi, 2000)
    for i in range(len(genome)):
        th, rho = genome[i]
        c, s = np.cos(th), np.sin(th)
        if abs(s) > 1e-6:
            y = (rho - x_range * c) / s
            ax.plot(x_range, y, "-", color="black", lw=LINE_WIDTH, alpha=1.0)
        else:
            ax.axvline(rho / c, color="black", lw=LINE_WIDTH, alpha=1.0)

    # Sanitize: No titles, No grids, No axes, No numbering
    ax.axis('off')
    ax.set_xlim(x_lo, x_hi)
    ax.set_ylim(y_lo, y_hi)
    ax.set_aspect("equal")

    os.makedirs(os.path.dirname(OUTPUT_IMAGE), exist_ok=True)
    fig.savefig(OUTPUT_IMAGE, dpi=DPI, bbox_inches="tight", pad_inches=0)
    plt.close(fig)
    print(f"[OEIS RENDER] Saved: {OUTPUT_IMAGE} (DPI={DPI})")


def main():
    genome = load_lines(TARGET_FILE)
    n = len(genome)
    print(f"[LOAD] Processing N={n} configuration for OEIS...")
    
    # We still perform the sweep to determine the correct viewport
    _, valid_triangles = count_triangles_internal(genome)
    print(f"[VERIFY] Topologically confirmed {len(valid_triangles)} triangles.")
    
    render_oeis(genome, valid_triangles)
    print("Minimalist OEIS illustration completed.")


def count_triangles_internal(genome):
    n = len(genome)
    intersections = solve_intersections(genome)
    valid = []
    for t in combinations(range(n), 3):
        verts = is_valid_triangle(genome, t, intersections)
        if verts is not None:
            valid.append((t, verts))
    return len(valid), valid


if __name__ == "__main__":
    main()
