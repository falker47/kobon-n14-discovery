import cupy as cp
import numpy as np
import os
import json
import sys
import glob
import time
from geometry_engine import calculate_fitness_batch

# CONFIGURATION
SOLUTIONS_DIR = "solutions"
BATCH_SIZE = 15000
TARGET_K = 54
N_LINES = 14

FROZEN_INDICES = list(range(0, 6))
MOBILE_INDICES = list(range(6, 14))
SIGMA_MUTATE = 1e-4  # Fine settling noise for hybrids

ELITE_COUNT = int(BATCH_SIZE * 0.05)  # 750
ELITE_CARRY = int(BATCH_SIZE * 0.10)  # 1500 carried forward

def load_gene_pool():
    """Load all solutions into a single GPU tensor."""
    files = glob.glob(os.path.join(SOLUTIONS_DIR, "*.json"))
    files.sort()
    
    all_genomes = []
    for f in files:
        with open(f, 'r') as fp:
            data = json.load(fp)
        lines = data['lines']
        lines.sort(key=lambda x: x['id'])
        genome = [[l['theta'], l['rho']] for l in lines]
        all_genomes.append(genome)
    
    pool_cpu = np.array(all_genomes, dtype=np.float64)  # (N_parents, 14, 2)
    pool_gpu = cp.asarray(pool_cpu)
    return pool_gpu, len(files)

def build_mask():
    """Mutation mask: 0=frozen, 1=mobile."""
    mask = np.zeros(N_LINES, dtype=np.float64)
    for i in MOBILE_INDICES:
        mask[i] = 1.0
    return cp.asarray(mask)

def crossover_from_pool(gene_pool, n_parents, n_children):
    """
    Multi-parent crossover. Fully vectorized.
    
    Core lines (0-5): Each child picks ONE random parent for all core lines.
    Mobile lines (6-13): Each child picks a DIFFERENT random parent PER line.
    
    Returns: (n_children, 14, 2) tensor.
    """
    children = cp.empty((n_children, N_LINES, 2), dtype=cp.float64)
    
    # --- Core Lines (0-5): One parent per child ---
    core_parent_idx = cp.random.randint(0, n_parents, n_children)  # (n_children,)
    # gene_pool[core_parent_idx] -> (n_children, 14, 2), then slice lines 0-5
    children[:, :6, :] = gene_pool[core_parent_idx][:, :6, :]
    
    # --- Mobile Lines (6-13): Different parent per line per child ---
    n_mobile = len(MOBILE_INDICES)
    # Shape: (n_children, n_mobile) — one parent index per mobile line per child
    mobile_parent_idx = cp.random.randint(0, n_parents, (n_children, n_mobile))
    
    # Gather: for each child i and mobile line j, pick gene_pool[mobile_parent_idx[i,j], MOBILE_INDICES[j], :]
    # Use advanced indexing
    child_idx = cp.arange(n_children)[:, None]  # (n_children, 1)
    mobile_line_idx = cp.array(MOBILE_INDICES)  # (8,)
    
    # gene_pool shape: (N_parents, 14, 2)
    # We need: gene_pool[mobile_parent_idx[i,j], mobile_line_idx[j], :] for all i,j
    # mobile_parent_idx: (n_children, 8)
    # mobile_line_idx:   (8,) broadcast to (1, 8)
    
    # Flatten for indexing
    flat_parents = mobile_parent_idx.ravel()  # (n_children * 8,)
    flat_lines = cp.tile(mobile_line_idx, n_children)  # (n_children * 8,)
    
    gathered = gene_pool[flat_parents, flat_lines, :]  # (n_children * 8, 2)
    gathered = gathered.reshape(n_children, n_mobile, 2)  # (n_children, 8, 2)
    
    children[:, 6:, :] = gathered
    
    return children

def save_grail(genome_cpu, k_value):
    filename = f"THE_HOLY_GRAIL_HYBRID_N{N_LINES}_K{k_value}.json"
    lines_data = []
    for i in range(len(genome_cpu)):
        lines_data.append({
            "id": i,
            "theta": float(genome_cpu[i, 0]),
            "rho": float(genome_cpu[i, 1])
        })
    out = {"n": N_LINES, "k": k_value, "lines": lines_data}
    with open(filename, "w") as f:
        json.dump(out, f, indent=2)
    return filename

def run_hybrid():
    # 1. Load Gene Pool
    gene_pool, n_parents = load_gene_pool()
    mask = build_mask()
    
    print(f"\n{'='*60}")
    print(f"  CAMPAIGN HYBRID — Cross-Pollination Mode")
    print(f"  Gene Pool: {n_parents} parents")
    print(f"  Batch:     {BATCH_SIZE} children/gen")
    print(f"  Frozen:    Lines 0-5  |  Mobile: Lines 6-13")
    print(f"  Sigma:     {SIGMA_MUTATE}")
    print(f"  Target:    K >= {TARGET_K}")
    print(f"{'='*60}\n")
    
    best_k_global = 0
    gen = 0
    t0 = time.time()
    
    # Initial population: crossover from pool
    population = crossover_from_pool(gene_pool, n_parents, BATCH_SIZE)
    
    while True:
        gen += 1
        
        # Evaluate fitness
        fitness = calculate_fitness_batch(population)
        current_max = int(cp.max(fitness))
        best_idx = int(cp.argmax(fitness))
        
        if current_max > best_k_global:
            best_k_global = current_max
            best_genome = population[best_idx].copy()
            elapsed = time.time() - t0
            print(f"[!!!] NEW RECORD K={best_k_global} at Gen {gen} ({elapsed:.1f}s)")
        
        # VICTORY CHECK
        if current_max >= TARGET_K:
            genome_cpu = cp.asnumpy(population[best_idx])
            saved = save_grail(genome_cpu, current_max)
            print(f"\n{'#'*60}")
            print(f"  !!! VICTORY !!! K={current_max} FOUND AT GEN {gen}")
            print(f"  Saved: {saved}")
            print(f"  Time:  {time.time() - t0:.2f}s")
            print(f"{'#'*60}\n")
            sys.exit(0)
        
        # Sort by fitness
        sorted_idx = cp.argsort(fitness)[::-1]
        sorted_pop = population[sorted_idx]
        
        # Preserve elites
        elites = sorted_pop[:ELITE_CARRY]
        
        # Generate new children via crossover from the ORIGINAL gene pool
        n_new = BATCH_SIZE - ELITE_CARRY
        new_children = crossover_from_pool(gene_pool, n_parents, n_new)
        
        # Apply fine mutation on mobile lines only
        noise_th = cp.random.normal(0, SIGMA_MUTATE, (n_new, N_LINES), dtype=cp.float64) * mask
        noise_rho = cp.random.normal(0, SIGMA_MUTATE, (n_new, N_LINES), dtype=cp.float64) * mask
        new_children[:, :, 0] += noise_th
        new_children[:, :, 1] += noise_rho
        new_children[:, :, 0] = new_children[:, :, 0] % cp.pi
        
        # Also mutate elites slightly (mobile only)
        elite_noise_th = cp.random.normal(0, SIGMA_MUTATE * 0.1, (ELITE_CARRY, N_LINES), dtype=cp.float64) * mask
        elite_noise_rho = cp.random.normal(0, SIGMA_MUTATE * 0.1, (ELITE_CARRY, N_LINES), dtype=cp.float64) * mask
        elites[:, :, 0] += elite_noise_th
        elites[:, :, 1] += elite_noise_rho
        elites[:, :, 0] = elites[:, :, 0] % cp.pi
        
        population = cp.concatenate([elites, new_children])
        
        # Progress
        if gen % 500 == 0:
            elapsed = time.time() - t0
            print(f"   Gen {gen:>7} | Best K: {best_k_global} | Time: {elapsed:.1f}s")

if __name__ == "__main__":
    run_hybrid()
