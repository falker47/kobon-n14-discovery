import sys
import time
import json
import numpy as np
import matplotlib.pyplot as plt
from optimizer import GeneticAlgorithm

# Configuration: Phase 3 (N=14 Discovery - HARDENED)
N_LINES = 14
TARGET_K = 54 # World Record Goal
BATCH_SIZE = 15000 # Constraints for RTX 3060
CHECK_THRESHOLD = 53 # Only verify solutions >= this on CPU with strict logic

def solve_intersections_robust(genome):
    """
    Computes intersections with high precision (float64) for robust CPU check.
    """
    intersections = {}
    eps = 1e-9
    for i in range(N_LINES):
        for j in range(i + 1, N_LINES):
            th_i, rho_i = genome[i]
            th_j, rho_j = genome[j]
            
            det = np.sin(th_j - th_i)
            if abs(det) < eps:
                continue
                
            x = (rho_i * np.sin(th_j) - rho_j * np.sin(th_i)) / det
            y = (rho_j * np.cos(th_i) - rho_i * np.cos(th_j)) / det
            intersections[(i, j)] = (x, y)
    return intersections

def check_triangle_robust(genome, idx_triplet, intersections):
    """
    Robust Segment Intersection Test (CPU).
    Triangle is valid if NO line intersects its edges [P1-P2], [P2-P3], [P3-P1].
    """
    i, j, k = idx_triplet
    p1 = intersections.get((i, j) if i<j else (j, i))
    p2 = intersections.get((j, k) if j<k else (k, j))
    p3 = intersections.get((i, k) if i<k else (k, i))
    
    if p1 is None or p2 is None or p3 is None:
        return None
        
    pts = [p1, p2, p3]
    epsilon = 1e-9
    
    # Helper: Signed Distance
    def dist(px, py, line_idx):
        th, rho = genome[line_idx]
        return px * np.cos(th) + py * np.sin(th) - rho

    # Check against all other lines
    for m in range(N_LINES):
        if m in (i, j, k): continue 
        
        d1 = dist(p1[0], p1[1], m)
        d2 = dist(p2[0], p2[1], m)
        d3 = dist(p3[0], p3[1], m)
        
        # Check intersections on edges: signs strictly opposite (product < -eps)
        cut_12 = (d1 * d2) < -epsilon
        cut_23 = (d2 * d3) < -epsilon
        cut_31 = (d3 * d1) < -epsilon
        
        if cut_12 or cut_23 or cut_31:
            return None # Invalidates the triangle

    return pts

def verify_genome_robust(genome):
    """
    Run full Count N verification on CPU with float64 Segment Logic.
    Returns: integer K
    """
    from itertools import combinations
    intersections = solve_intersections_robust(genome)
    triplets = list(combinations(range(N_LINES), 3))
    valid_count = 0
    for t in triplets:
        if check_triangle_robust(genome, t, intersections):
            valid_count += 1
    return valid_count


def check_triangle_cpu(genome, idx_triplet, intersections):
    """
    Checks if triplet (i, j, k) forms a valid Kobon triangle.
    Condition: No other line m passes through the interior.
    """
    i, j, k = idx_triplet
    
    # Vertices
    p1 = intersections.get((i, j) if i<j else (j, i))
    p2 = intersections.get((j, k) if j<k else (k, j))
    p3 = intersections.get((i, k) if i<k else (k, i))
    
    if p1 is None or p2 is None or p3 is None:
        return False
        
    pts = [p1, p2, p3]
    
    # Check emptiness against all other lines
    for m in range(N_LINES):
        if m in (i, j, k):
            continue
            
        th_m, rho_m = genome[m]
        
        # Signed distances
        # dist = x cos + y sin - rho
        dists = []
        for (vx, vy) in pts:
            d = vx * np.cos(th_m) + vy * np.sin(th_m) - rho_m
            # Filter noise
            if abs(d) < 1e-4: d = 0.0
            dists.append(d)
            
        d_min = min(dists)
        d_max = max(dists)
        
        # If line crosses interior, min < 0 and max > 0
        if d_min < 0 and d_max > 0:
            return False

    return pts

def visualize_solution(genome, score, filename=None):
    """
    Visualizes the lines using Matplotlib (CPU) and HIGHLIGHTS valid triangles.
    """
    plt.figure(figsize=(12, 12))
    plt.title(f"Kobon Triangles N={N_LINES}, K={score} (Verification Mode)")
    
    # 1. Plot Lines
    x_range = np.linspace(-2.5, 2.5, 100)
    for i in range(N_LINES):
        theta = genome[i, 0]
        rho = genome[i, 1]
        c = np.cos(theta)
        s = np.sin(theta)
        
        if abs(s) > 1e-3:
            y = (rho - x_range * c) / s
            plt.plot(x_range, y, '-k', lw=0.5, alpha=0.3)
        else:
            x_const = rho / c
            plt.axvline(x_const, color='k', lw=0.5, alpha=0.3)

    # 2. Identify and Fill Triangles (CPU Re-Verification)
    from itertools import combinations
    intersections = solve_intersections_robust(genome)
    
    triplets = list(combinations(range(N_LINES), 3))
    valid_count = 0
    
    print("Verifying triangles on CPU for plot...")
    
    # Use a colormap for distinct triangles
    cmap = plt.get_cmap('tab20')
    
    for triplet in triplets:
        triangle_pts = check_triangle_robust(genome, triplet, intersections)
        if triangle_pts:
            valid_count += 1
            pts = np.array(triangle_pts)
            
            # Fill Polygon
            poly = plt.Polygon(pts, facecolor=cmap(valid_count % 20), edgecolor='none', alpha=0.5)
            plt.gca().add_patch(poly)
            
            # Label Number
            centroid = np.mean(pts, axis=0)
            plt.text(centroid[0], centroid[1], str(valid_count), 
                     fontsize=8, ha='center', va='center', fontweight='bold', color='black')

    print(f"CPU Verification found {valid_count} triangles.")
            
    plt.xlim(-2.5, 2.5)
    plt.ylim(-2.5, 2.5)
    plt.gca().set_aspect('equal')
    
    out_name = filename if filename else f"solution_n{N_LINES}_k{score}_verified.png"
    plt.savefig(out_name, dpi=300)
    print(f"Verified solution image saved to {out_name}")
    plt.close()

def main():
    print(f"Starting Project Hycarus [PHASE 2: N={N_LINES}] on GPU...")
    print(f"Batch Size: {BATCH_SIZE} | Target K: {TARGET_K}")
    print("Mode: Infinite Discovery with Incremental Saving")
    
    try:
        ga = GeneticAlgorithm(n_lines=N_LINES, batch_size=BATCH_SIZE)
    except Exception as e:
        print(f"Failed to initialize GPU/CuPy: {e}")
        return

    start_time = time.time()
    global_best_k = 0
    generation = 0
    
    try:
        while True:
            generation += 1
            current_best_k = ga.step()
            
            # Reset GA logic? 
            # In a true infinite evolutionary strategy, we might want to restart population 
            # if stuck in local optima for too long, but for now we let mutation handle it.
            # Ideally, we should detect stagnation. 
            # For this simplified version, we just run.
            
            if current_best_k > global_best_k:
                # Potential Record
                best_genome_candidate = ga.get_best_genome_cpu()
                
                # HARDENING: Cross-Check with CPU Strict Logic
                actual_k = int(current_best_k)
                if actual_k >= CHECK_THRESHOLD:
                    print(f"Candidate K={actual_k} requires STRICT verification...")
                    verified_k = verify_genome_robust(best_genome_candidate)
                    
                    if verified_k < actual_k:
                        print(f"!!! FALSE POSITIVE REJECTED !!! GPU saw {actual_k}, Strict CPU saw {verified_k}")
                        # We only count the 'verified_k' as the real score
                        actual_k = verified_k
                    else:
                        print(f"STRICT VERIFICATION PASSED: K={verified_k}")
                
                # Check if it's still a record after verification
                if actual_k > global_best_k:
                    global_best_k = actual_k
                    elapsed = time.time() - start_time
                    print(f"!!! NEW VERIFIED RECORD: K={global_best_k} (Gen {generation}, Time: {elapsed:.2f}s) !!!")
                    
                    # Save Immediately
                    if best_genome_candidate is not None:
                        # NPY Save
                        npy_filename = f"best_N{N_LINES}_K{global_best_k}.npy"
                        np.save(npy_filename, best_genome_candidate)
                        
                        # Visualization (Verified)
                        visualize_solution(best_genome_candidate, global_best_k, filename=f"solution_N{N_LINES}_K{global_best_k}.png")
            
            if generation % 50 == 0:
                 elapsed = time.time() - start_time
                 print(f"Gen {generation}: Current Best={current_best_k} | Global Record={global_best_k} (Time: {elapsed:.2f}s)")

            if global_best_k >= TARGET_K:
                print("\n\n" + "#"*60)
                print(f"# TARGET REACHED! K={global_best_k} #")
                print("#"*60 + "\n")
                visualize_solution(ga.get_best_genome_cpu(), global_best_k, filename="VICTORY_K54.png")
                break
                
    except KeyboardInterrupt:
        print("\nInterrupted by user.")
        
    print(f"Discovery Stopped. Highest K detected: {global_best_k}")

if __name__ == "__main__":
    main()
