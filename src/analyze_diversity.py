import os
import glob
import json
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

SOLUTIONS_DIR = "solutions"
N_LINES = 14
VARIANCE_THRESHOLD = 1e-4

def load_solutions():
    files = glob.glob(os.path.join(SOLUTIONS_DIR, "*.json"))
    files.sort()
    
    all_rhos = []
    
    for f in files:
        with open(f, 'r') as fp:
            data = json.load(fp)
        
        lines = data['lines']
        # Extract |rho| values, sorted ascending (order-invariant)
        rhos = sorted([abs(line['rho']) for line in lines])
        all_rhos.append(rhos)
    
    return np.array(all_rhos), len(files)

def analyze(matrix, n_solutions):
    print(f"\n{'='*60}")
    print(f"  DIVERSITY ANALYSIS — {n_solutions} solutions, {N_LINES} lines")
    print(f"{'='*60}\n")
    
    # Per-index variance across all solutions
    variances = np.var(matrix, axis=0)
    means = np.mean(matrix, axis=0)
    stds = np.std(matrix, axis=0)
    
    print(f"{'Index':>5}  {'Mean |ρ|':>10}  {'Std':>10}  {'Variance':>12}  {'Status'}")
    print(f"{'-'*5}  {'-'*10}  {'-'*10}  {'-'*12}  {'-'*10}")
    
    core_lines = []
    mobile_lines = []
    
    for i in range(N_LINES):
        status = "CORE" if variances[i] < VARIANCE_THRESHOLD else "MOBILE"
        marker = "🔒" if status == "CORE" else "🔀"
        print(f"{i:>5}  {means[i]:>10.6f}  {stds[i]:>10.6f}  {variances[i]:>12.8f}  {marker} {status}")
        
        if status == "CORE":
            core_lines.append(i)
        else:
            mobile_lines.append(i)
    
    print(f"\n{'='*60}")
    print(f"  SUMMARY")
    print(f"{'='*60}")
    print(f"  Core Lines (Frozen):  {len(core_lines)}/14  {core_lines}")
    print(f"  Mobile Lines (Free):  {len(mobile_lines)}/14  {mobile_lines}")
    print(f"  Overall Diversity:    {np.mean(variances):.8f}")
    print()
    
    return variances

def plot_skeleton(matrix, variances):
    fig, ax = plt.subplots(figsize=(12, 6))
    
    # Plot each solution as a semi-transparent line
    for row in matrix:
        ax.plot(range(N_LINES), row, color='black', alpha=0.05, linewidth=0.8)
    
    # Overlay mean as a bold red line
    means = np.mean(matrix, axis=0)
    ax.plot(range(N_LINES), means, color='red', linewidth=2, label='Mean |ρ|', zorder=10)
    
    # Mark core vs mobile
    for i in range(N_LINES):
        color = 'green' if variances[i] < VARIANCE_THRESHOLD else 'orange'
        ax.scatter(i, means[i], color=color, s=60, zorder=11, edgecolors='black', linewidths=0.5)
    
    ax.set_xlabel("Sorted Line Index", fontsize=12)
    ax.set_ylabel("|ρ| (Absolute Distance)", fontsize=12)
    ax.set_title(f"Kobon N=14 Solution Skeleton ({len(matrix)} solutions)", fontsize=14)
    ax.set_xticks(range(N_LINES))
    ax.legend(loc='upper left')
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig("diversity_analysis.png", dpi=150)
    print(f"[OK] Figure saved -> diversity_analysis.png")

if __name__ == "__main__":
    matrix, n_solutions = load_solutions()
    variances = analyze(matrix, n_solutions)
    plot_skeleton(matrix, variances)
