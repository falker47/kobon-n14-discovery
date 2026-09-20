"""
ANIMATE_GRAIL.PY — Dynamic Construction of the K=54 Record
============================================================
Standalone script. No GPU dependencies.
Produces:
  - HOLY_GRAIL_K54_CONSTRUCTION.mp4  (ffmpeg, 60 FPS)
  - HOLY_GRAIL_K54_CONSTRUCTION.gif  (pillow fallback)
"""
import os
import sys
import json
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from itertools import combinations

# ─── CONFIGURATION ──────────────────────────────────────────────────────────
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
TARGET_FILE  = os.path.join(PROJECT_ROOT, "data", "solutions_run2", "N14_K54_3c335303.json")
OUTPUT_DIR   = os.path.join(PROJECT_ROOT, "assets")

FPS          = 60
LINE_FRAMES  = 14          # one line per frame
PAUSE_SECS   = 1.0         # hold full grid
TRI_FRAMES   = 54          # one triangle per frame
TAIL_SECS    = 2.0         # hold final result

EPSILON      = 1e-9
# ────────────────────────────────────────────────────────────────────────────


def load_lines(filepath: str) -> np.ndarray:
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


def compute_viewport(valid_triangles, pad_frac=0.05):
    all_verts = []
    for _, verts in valid_triangles:
        all_verts.extend(verts)
    all_verts = np.array(all_verts)
    x_min, x_max = all_verts[:, 0].min(), all_verts[:, 0].max()
    y_min, y_max = all_verts[:, 1].min(), all_verts[:, 1].max()
    px = (x_max - x_min) * pad_frac
    py = (y_max - y_min) * pad_frac
    return x_min - px, x_max + px, y_min - py, y_max + py


def main():
    print("=" * 60)
    print("  ANIMATE_GRAIL — K=54 Construction Animation")
    print("=" * 60)

    genome = load_lines(TARGET_FILE)
    n = len(genome)
    print(f"[LOAD] {n} lines from {os.path.basename(TARGET_FILE)}")

    valid_triangles = find_valid_triangles(genome)
    k = len(valid_triangles)
    print(f"[VERIFY] {k} valid triangles confirmed.")

    x_lo, x_hi, y_lo, y_hi = compute_viewport(valid_triangles)
    print(f"[VIEWPORT] X=[{x_lo:.3f}, {x_hi:.3f}]  Y=[{y_lo:.3f}, {y_hi:.3f}]")

    # ── Precompute line plotting data ──
    x_range = np.linspace(x_lo, x_hi, 1000)
    line_data = []
    for i in range(n):
        th, rho = genome[i]
        c, s = np.cos(th), np.sin(th)
        if abs(s) > 1e-6:
            ys = (rho - x_range * c) / s
            mask = (ys >= y_lo - 1) & (ys <= y_hi + 1)
            line_data.append(("line", x_range[mask], ys[mask]))
        else:
            xv = rho / c
            line_data.append(("vline", xv, None))

    # ── Frame budget ──
    pause_frames = int(PAUSE_SECS * FPS)
    tail_frames  = int(TAIL_SECS * FPS)
    total_frames = LINE_FRAMES + pause_frames + TRI_FRAMES + tail_frames
    print(f"[ANIM] {total_frames} frames @ {FPS} FPS "
          f"({total_frames / FPS:.1f}s total)")

    # ── Setup figure ──
    fig, ax = plt.subplots(figsize=(16, 9), dpi=300)
    ax.set_xlim(x_lo, x_hi)
    ax.set_ylim(y_lo, y_hi)
    ax.set_aspect("equal")
    ax.set_facecolor("#0a0a0a")
    fig.patch.set_facecolor("#0a0a0a")
    for spine in ax.spines.values():
        spine.set_visible(False)
    ax.set_xticks([])
    ax.set_yticks([])

    # Title text (will update dynamically)
    title_text = ax.text(
        0.5, 0.97, "", transform=ax.transAxes,
        ha="center", va="top",
        fontsize=14, fontweight="bold",
        color="white", family="monospace",
    )

    # Counter text (bottom-right)
    counter_text = ax.text(
        0.98, 0.03, "", transform=ax.transAxes,
        ha="right", va="bottom",
        fontsize=12, fontweight="bold",
        color="#cccccc", family="monospace",
    )

    # Persistent artists
    drawn_lines = []
    drawn_polys = []

    def init():
        title_text.set_text("")
        counter_text.set_text("")
        return [title_text, counter_text]

    def update(frame):
        artists = []

        # ── Phase 1: Draw lines one by one ──
        if frame < LINE_FRAMES:
            idx = frame
            title_text.set_text(f"Drawing Line {idx + 1} / {n}")
            counter_text.set_text("")

            kind, xd, yd = line_data[idx]
            if kind == "line":
                ln, = ax.plot(xd, yd, "-", color="#4fc3f7", lw=0.4, alpha=0.8)
            else:
                ln = ax.axvline(xd, color="#4fc3f7", lw=0.4, alpha=0.8)
            drawn_lines.append(ln)
            artists.append(ln)

        # ── Phase 2: Pause on full grid ──
        elif frame < LINE_FRAMES + pause_frames:
            title_text.set_text(f"N = {n}  Lines  •  Searching Triangles...")
            counter_text.set_text(f"{len(drawn_polys)} / {k} triangles")

        # ── Phase 3: Overlay triangles one by one ──
        elif frame < LINE_FRAMES + pause_frames + TRI_FRAMES:
            tri_idx = frame - LINE_FRAMES - pause_frames
            _, verts = valid_triangles[tri_idx]
            pts = np.array(verts)

            poly = plt.Polygon(
                pts,
                facecolor="crimson",
                edgecolor="none",
                alpha=0.4,
            )
            ax.add_patch(poly)
            drawn_polys.append(poly)
            artists.append(poly)

            title_text.set_text(f"Kobon Triangle #{tri_idx + 1}")
            counter_text.set_text(f"{tri_idx + 1} / {k} triangles")

        # ── Phase 4: Tail hold ──
        else:
            title_text.set_text(f"K(14) = {k}   •   CONFIRMED")
            counter_text.set_text(f"{k} / {k} triangles")

        artists.extend([title_text, counter_text])
        return artists

    anim = FuncAnimation(
        fig,
        update,
        frames=total_frames,
        init_func=init,
        blit=False,
        repeat=False,
    )

    os.makedirs(OUTPUT_DIR, exist_ok=True)

    # ── Try MP4 (ffmpeg) first, fallback to GIF (pillow) ──
    mp4_path = os.path.join(OUTPUT_DIR, "HOLY_GRAIL_K54_CONSTRUCTION.mp4")
    gif_path = os.path.join(OUTPUT_DIR, "HOLY_GRAIL_K54_CONSTRUCTION.gif")

    try:
        print(f"[EXPORT] Saving MP4 -> {mp4_path}")
        anim.save(mp4_path, writer="ffmpeg", fps=FPS)
        print(f"[DONE] {mp4_path}")
    except Exception as e:
        print(f"[WARN] ffmpeg unavailable ({e}). Falling back to GIF...")
        try:
            print(f"[EXPORT] Saving GIF -> {gif_path}")
            anim.save(gif_path, writer="pillow", fps=min(FPS, 30))
            print(f"[DONE] {gif_path}")
        except Exception as e2:
            print(f"[ERROR] GIF export also failed: {e2}")
            sys.exit(1)

    plt.close(fig)
    print("\nAnimation complete.")


if __name__ == "__main__":
    main()
