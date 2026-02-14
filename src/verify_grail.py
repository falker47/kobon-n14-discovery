"""
VERIFY_GRAIL.PY — Strict CPU Validation for Candidate K=54
===========================================================
Standalone script. Zero GPU dependencies.
Loads a candidate JSON, counts Kobon triangles with float64 segment-intersection
logic, and renders a high-fidelity diagram if validation passes.
"""
import os
import sys
import json
import numpy as np
import matplotlib
matplotlib.use("Agg")          # headless backend — no GUI needed
import matplotlib.pyplot as plt
from itertools import combinations

# ─── CONFIGURATION ──────────────────────────────────────────────────────────
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
TARGET_FILE  = os.path.join(PROJECT_ROOT, "data", "solutions_run2", "N14_K54_3c335303.json")
OUTPUT_IMAGE = os.path.join(PROJECT_ROOT, "assets", "Kobon_N14_K54.png")

EPSILON      = 1e-9          # parallelism / cut tolerance
RENDER_DPI   = 600
FIG_SIZE     = (20, 20)
# ────────────────────────────────────────────────────────────────────────────


def load_lines(filepath: str) -> np.ndarray:
    """Load lines from JSON -> (N, 2) float64 array of [theta, rho]."""
    if not os.path.isfile(filepath):
        raise FileNotFoundError(f"CRITICAL: Target file not found: {filepath}")

    with open(filepath, "r") as f:
        data = json.load(f)

    lines = sorted(data["lines"], key=lambda x: x["id"])
    genome = np.array([[l["theta"], l["rho"]] for l in lines], dtype=np.float64)
    return genome


def solve_intersections(genome: np.ndarray) -> dict:
    """Compute all pairwise intersections via Cramer's rule (float64)."""
    n = len(genome)
    intersections = {}
    for i in range(n):
        for j in range(i + 1, n):
            th_i, rho_i = genome[i]
            th_j, rho_j = genome[j]
            det = np.sin(th_j - th_i)
            if abs(det) < EPSILON:
                continue                 # parallel pair
            x = (rho_i * np.sin(th_j) - rho_j * np.sin(th_i)) / det
            y = (rho_j * np.cos(th_i) - rho_i * np.cos(th_j)) / det
            intersections[(i, j)] = (x, y)
    return intersections


def is_valid_triangle(genome, triplet, intersections) -> list | None:
    """
    Segment-intersection test (CPU, float64).
    Returns list of 3 vertex tuples if valid, else None.
    """
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

        # Strict edge-cut: product < -epsilon means the line
        # passes strictly between two vertices on that edge.
        if (d1 * d2) < -EPSILON or (d2 * d3) < -EPSILON or (d3 * d1) < -EPSILON:
            return None

    return [p1, p2, p3]


def count_triangles(genome):
    """Full CPU sweep. Returns (count, list_of_valid_triangles_with_vertices)."""
    n = len(genome)
    intersections = solve_intersections(genome)
    triplets = list(combinations(range(n), 3))

    valid = []
    for t in triplets:
        verts = is_valid_triangle(genome, t, intersections)
        if verts is not None:
            valid.append((t, verts))
    return len(valid), valid


def render(genome, valid_triangles, k_verified):
    """Micro-rendering: dynamic bounding box, DPI=1200, hairline lines."""
    n = len(genome)

    # ── 1. Dynamic Bounding Box from ALL triangle vertices ──
    all_verts = []
    for _, verts in valid_triangles:
        all_verts.extend(verts)
    all_verts = np.array(all_verts)  # (V, 2)

    x_min, x_max = all_verts[:, 0].min(), all_verts[:, 0].max()
    y_min, y_max = all_verts[:, 1].min(), all_verts[:, 1].max()
    pad_x = (x_max - x_min) * 0.05
    pad_y = (y_max - y_min) * 0.05
    x_lo, x_hi = x_min - pad_x, x_max + pad_x
    y_lo, y_hi = y_min - pad_y, y_max + pad_y

    print(f"[RENDER] Viewport: X=[{x_lo:.4f}, {x_hi:.4f}]  Y=[{y_lo:.4f}, {y_hi:.4f}]")
    print(f"[RENDER] {len(all_verts)} vertices from {len(valid_triangles)} triangles")

    # ── 2. Create massive figure ──
    fig, ax = plt.subplots(figsize=(30, 30))
    # ax.set_title(
    #     f"HOLY GRAIL — N={n}  K={k_verified}  [CPU STRICT VALIDATION]",
    #     fontsize=28, fontweight="bold", pad=24,
    # )

    # ── 3. Draw lines (hairline) ──
    x_range = np.linspace(x_lo, x_hi, 1000)
    for i in range(n):
        th, rho = genome[i]
        c, s = np.cos(th), np.sin(th)
        if abs(s) > 1e-6:
            y = (rho - x_range * c) / s
            ax.plot(x_range, y, "-", color="#222222", lw=0.1, alpha=0.7)
        else:
            ax.axvline(rho / c, color="#222222", lw=0.1, alpha=0.7)

    # ── 4. Fill valid triangles (uniform red) ──
    for idx, (triplet, verts) in enumerate(valid_triangles):
        pts = np.array(verts)
        poly = plt.Polygon(
            pts,
            facecolor="red",
            edgecolor="none",
            alpha=0.3,
        )
        ax.add_patch(poly)
        cx, cy = np.mean(pts, axis=0)
        ax.text(
            cx, cy, str(idx + 1),
            fontsize=3, ha="center", va="center",
            fontweight="bold", color="black",
        )

    # ── 5. Draw intersection points ──
    ax.scatter(
        all_verts[:, 0], all_verts[:, 1],
        s=1, c="blue", zorder=5, marker=".", linewidths=0,
    )

    ax.set_xlim(x_lo, x_hi)
    ax.set_ylim(y_lo, y_hi)
    ax.set_aspect("equal")
    ax.grid(True, alpha=0.1, linewidth=0.1)

    os.makedirs(os.path.dirname(OUTPUT_IMAGE), exist_ok=True)
    fig.savefig(OUTPUT_IMAGE, dpi=1200, bbox_inches="tight")
    plt.close(fig)
    print(f"[RENDER] Saved: {OUTPUT_IMAGE}  (DPI=1200, figsize=30x30)")


# ─── MAIN ───────────────────────────────────────────────────────────────────
def main():
    banner = "=" * 60
    print(banner)
    print("  VERIFY_GRAIL — Strict CPU Validation")
    print(f"  Target: {os.path.basename(TARGET_FILE)}")
    print(banner)

    genome = load_lines(TARGET_FILE)
    n = len(genome)
    print(f"[LOAD] {n} lines loaded (float64).")

    print("[VERIFY] Running full C({n},3) = {0} triplet sweep...".format(
        len(list(combinations(range(n), 3))), n=n
    ))

    k_verified, valid_triangles = count_triangles(genome)

    print()
    print(banner)
    print(f"  [CPU STRICT VALIDATION]: K = {k_verified}")
    print(banner)

    if k_verified >= 54:
        print(f"█   K = {k_verified}  CONFIRMED — THE GRAIL IS REAL            █")
    else:
        print(f"\n⚠  GPU HALLUCINATION DETECTED: True K = {k_verified}, not 54.")
        print("   Epsilon ghost intersections inflated the GPU count.\n")

    # Render regardless — the diagram is useful even for K < 54
    print("[RENDER] Generating high-fidelity diagram...")
    render(genome, valid_triangles, k_verified)

    print("\nDone.")


if __name__ == "__main__":
    main()
