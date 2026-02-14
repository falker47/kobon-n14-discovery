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
MAX_GENERATIONS = 3000
TARGET_K = 54
N_LINES = 14

# FREEZE MASK: 0=Frozen, 1=Mutable
FROZEN_INDICES = list(range(0, 6))   # Lines 0-5: CORE (locked)
MOBILE_INDICES = list(range(6, 14))  # Lines 6-13: MOBILE (optimized)
SIGMA = 1e-3  # Aggressive exploration on mobile lines

ELITE_COUNT = int(BATCH_SIZE * 0.05)

def load_solution(filename):
    with open(filename, 'r') as f:
        data = json.load(f)
    lines = data['lines']
    lines.sort(key=lambda x: x['id'])
    genome = [[line['theta'], line['rho']] for line in lines]
    return np.array(genome, dtype=np.float64), data['n']

def build_mask():
    """Create (N_LINES,) mask: 0.0 for frozen, 1.0 for mobile."""
    mask = np.zeros(N_LINES, dtype=np.float64)
    for i in MOBILE_INDICES:
        mask[i] = 1.0
    return cp.asarray(mask)

def surgical_step(population, mask, elite_count):
    """One generation of masked GA. Returns (new_population, current_best_k, best_genome)."""
    fitness = calculate_fitness_batch(population)
    
    current_max = int(cp.max(fitness))
    best_idx = int(cp.argmax(fitness))
    best_genome = population[best_idx].copy()
    
    sorted_indices = cp.argsort(fitness)[::-1]
    sorted_pop = population[sorted_indices]
    
    elites = sorted_pop[:elite_count]
    n_offspring = BATCH_SIZE - elite_count
    
    # Selection: top half
    top_half = BATCH_SIZE // 2
    parents_idx = cp.random.randint(0, top_half, (n_offspring, 2))
    p1 = sorted_pop[parents_idx[:, 0]]
    p2 = sorted_pop[parents_idx[:, 1]]
    
    # Crossover
    crossover_mask = cp.random.rand(n_offspring, N_LINES, 1) < 0.5
    offspring = cp.where(crossover_mask, p1, p2)
    
    # Mutation (MASKED)
    # mask shape (N_LINES,) -> broadcast to (n_offspring, N_LINES)
    mutate_prob = cp.random.rand(n_offspring, N_LINES) < 0.05  # 5% mutation rate
    
    noise_th = cp.random.normal(0, SIGMA, (n_offspring, N_LINES), dtype=cp.float64)
    noise_rho = cp.random.normal(0, SIGMA, (n_offspring, N_LINES), dtype=cp.float64)
    
    # Apply mask: only mobile lines get noise
    noise_th *= mask  # broadcast (N_LINES,) across batch
    noise_rho *= mask
    
    offspring[:, :, 0] += mutate_prob * noise_th
    offspring[:, :, 1] += mutate_prob * noise_rho
    offspring[:, :, 0] = offspring[:, :, 0] % cp.pi
    
    new_population = cp.concatenate([elites, offspring])
    return new_population, current_max, best_genome

def save_grail(genome_cpu, k_value):
    filename = f"THE_HOLY_GRAIL_SURGICAL_N{N_LINES}_K{k_value}.json"
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

def run_campaign():
    files = glob.glob(os.path.join(SOLUTIONS_DIR, "*.json"))
    files.sort()
    total = len(files)
    
    mask = build_mask()
    
    print(f"\n{'='*60}")
    print(f"  CAMPAIGN SURGICAL — {total} files")
    print(f"  Frozen: {len(FROZEN_INDICES)} lines ({FROZEN_INDICES})")
    print(f"  Mobile: {len(MOBILE_INDICES)} lines ({MOBILE_INDICES})")
    print(f"  Sigma:  {SIGMA}  |  Budget: {MAX_GENERATIONS} gens/file")
    print(f"{'='*60}\n")
    
    for idx, filepath in enumerate(files):
        basename = os.path.basename(filepath)
        print(f"\n[Surgical Strike] File {idx+1}/{total} | {basename}")
        
        try:
            seed_cpu, n = load_solution(filepath)
            seed_gpu = cp.asarray(seed_cpu, dtype=cp.float64)
            
            # Clone seed -> batch
            population = cp.tile(seed_gpu, (BATCH_SIZE, 1, 1))
            
            # Initial noise on MOBILE lines only
            init_noise_th = cp.random.normal(0, SIGMA, (BATCH_SIZE, N_LINES), dtype=cp.float64) * mask
            init_noise_rho = cp.random.normal(0, SIGMA, (BATCH_SIZE, N_LINES), dtype=cp.float64) * mask
            population[:, :, 0] += init_noise_th
            population[:, :, 1] += init_noise_rho
            population[:, :, 0] = population[:, :, 0] % cp.pi
            
            best_k = 0
            t0 = time.time()
            
            for gen in range(1, MAX_GENERATIONS + 1):
                population, current_k, best_genome = surgical_step(population, mask, ELITE_COUNT)
                
                if current_k > best_k:
                    best_k = current_k
                
                if gen % 2500 == 0:
                    print(f"   Gen {gen:>6}/{MAX_GENERATIONS} | Best: {best_k}", end="\r")
                
                if current_k >= TARGET_K:
                    elapsed = time.time() - t0
                    genome_cpu = cp.asnumpy(best_genome)
                    saved = save_grail(genome_cpu, current_k)
                    
                    print(f"\n\n{'#'*60}")
                    print(f"  !!! VICTORY !!! K={current_k} FOUND AT GEN {gen}")
                    print(f"  Source: {basename}")
                    print(f"  Saved:  {saved}")
                    print(f"  Time:   {elapsed:.2f}s")
                    print(f"{'#'*60}\n")
                    sys.exit(0)
            
            elapsed = time.time() - t0
            print(f"   [Done] Max K={best_k} | Time: {elapsed:.1f}s                    ")
            
            # Cleanup VRAM
            del population, best_genome
            cp.get_default_memory_pool().free_all_blocks()
            
        except Exception as e:
            print(f"   [ERROR] {e}")
            continue
    
    print(f"\n[Campaign Complete] No K={TARGET_K} found. All {total} files exhausted.")

if __name__ == "__main__":
    run_campaign()
