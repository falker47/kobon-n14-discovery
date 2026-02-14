"""
OPTIMIZER TABOO — Anti-Gravity Protocol
Bans the known K=53 core skeleton to force discovery of new topological families.
"""
import cupy as cp
import numpy as np
import os
import json
import sys
import glob
import time
import hashlib
import shutil
from geometry_engine import calculate_fitness_batch

# CONFIGURATION
BATCH_SIZE = 10000
N_LINES = 14
TARGET_K = 54

# TABOO PARAMETERS
TABOO_THRESHOLD = 0.05   # MSE below this = "falling into the black hole"
TABOO_PENALTY = 15       # Fitness points to subtract

# DEEP TABOO PARAMETERS
FAST_FAIL_LIMIT = 3000   # Gens before kill if K < 53
FAST_FAIL_K = 53
STAGNATION_LIMIT = 3000  # Gens before hard reset
MUTATION_RATE = 0.01
ELITE_RATIO = 0.05

# ─── TABOO CORE EXTRACTION ─────────────────────────────────────────────────

# Hardcoded Alpha Core path — relative to the project root (one level up from src/).
ALPHA_CORE_PATH = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "ALPHA_CORE_K53.json")

def extract_taboo_core():
    """Load ALPHA_CORE_K53.json and extract Core Lines (0-5) as the taboo tensor."""
    if not os.path.isfile(ALPHA_CORE_PATH):
        raise FileNotFoundError("CRITICAL: ALPHA_CORE_K53.json missing. Cannot start Taboo Search.")
    
    with open(ALPHA_CORE_PATH, 'r') as f:
        data = json.load(f)
    
    lines = data['lines']
    lines.sort(key=lambda x: x['id'])
    
    # Extract core lines 0-5: shape (6, 2) = [theta, rho]
    core = []
    for i in range(6):
        core.append([lines[i]['theta'], lines[i]['rho']])
    
    taboo_cpu = np.array(core, dtype=np.float64)
    taboo_gpu = cp.asarray(taboo_cpu)  # (6, 2)
    
    print(f"[TABOO] Loaded taboo core from: {ALPHA_CORE_PATH}")
    print(f"[TABOO] Core shape: {taboo_gpu.shape}")
    print(f"[TABOO] Threshold: {TABOO_THRESHOLD} | Penalty: -{TABOO_PENALTY}")
    return taboo_gpu

# ─── ARCHIVE OLD SOLUTIONS ─────────────────────────────────────────────────

def archive_solutions():
    """Move existing solutions from data/solutions_run2 to archive folder."""
    base_dir = "data/solutions_run2"
    if not os.path.exists(base_dir):
        os.makedirs(base_dir, exist_ok=True)
        return
    
    files = glob.glob(os.path.join(base_dir, "*.json"))
    if not files:
        return
    
    timestamp = time.strftime("%Y%m%d_%H%M%S")
    archive_dir = f"data/solutions_archive_{timestamp}"
    os.makedirs(archive_dir, exist_ok=True)
    
    for f in files:
        try:
            shutil.move(f, os.path.join(archive_dir, os.path.basename(f)))
        except Exception as e:
            print(f"[ARCHIVE ERROR] Could not move {f}: {e}")
    
    print(f"[ARCHIVE] Moved {len(files)} files -> {archive_dir}/")
    sys.stdout.flush()

# ─── TABOO OPTIMIZER ───────────────────────────────────────────────────────

class TabooOptimizer:
    def __init__(self, n_lines, taboo_core, batch_size=BATCH_SIZE):
        self.n = n_lines
        self.batch_size = batch_size
        self.elite_count = int(batch_size * ELITE_RATIO)
        
        self.taboo_core = taboo_core  # (6, 2) or None
        
        self.population = self._initialize_population()
        self.best_fitness = 0
        self.best_genome = None
        
        self.generations_without_improvement = 0
        self.fast_fail_counter = 0
        self.consecutive_stagnations = 0
        
        # Deduplication
        self.seen_hashes = set()
        self.variant_counter = 0
        
        # Taboo tracking
        self.total_taboo_hits = 0
    
    def _initialize_population(self):
        base_thetas = cp.arange(self.n, dtype=cp.float64) * (cp.pi / self.n)
        base_rhos = cp.ones(self.n, dtype=cp.float64)
        
        thetas_batch = cp.tile(base_thetas, (self.batch_size, 1))
        rhos_batch = cp.tile(base_rhos, (self.batch_size, 1))
        
        noise_th = cp.random.uniform(-0.2, 0.2, (self.batch_size, self.n), dtype=cp.float64)
        noise_rho = cp.random.uniform(-0.5, 0.5, (self.batch_size, self.n), dtype=cp.float64)
        
        thetas_batch += noise_th
        rhos_batch += noise_rho
        thetas_batch = thetas_batch % cp.pi
        
        return cp.stack([thetas_batch, rhos_batch], axis=-1)
    
    def _apply_taboo_penalty(self, fitness):
        """
        Vectorized novelty penalty.
        Penalize individuals whose core lines (0-5) are too close to the taboo skeleton.
        """
        if self.taboo_core is None:
            return fitness
        
        # Extract core lines from all individuals: (Batch, 6, 2)
        batch_core = self.population[:, :6, :]
        
        # taboo_core shape: (6, 2) -> broadcast to (1, 6, 2)
        # MSE per individual: mean over lines and coords
        diff = batch_core - self.taboo_core[None, :, :]  # (Batch, 6, 2)
        mse = cp.mean(diff ** 2, axis=(1, 2))             # (Batch,)
        
        # Penalize those falling into the black hole
        falling = mse < TABOO_THRESHOLD
        n_penalized = int(cp.sum(falling))
        self.total_taboo_hits += n_penalized
        
        fitness = fitness.astype(cp.float64)
        fitness[falling] -= TABOO_PENALTY
        
        return fitness, n_penalized
    
    def _trigger_hard_reset(self):
        """Full population wipe. Always hard."""
        print("[!!!] FULL WIPE: Generating 100% fresh random population.")
        random_thetas = cp.random.uniform(0, cp.pi, (self.batch_size, self.n), dtype=cp.float64)
        random_rhos = cp.random.uniform(-2.0, 2.0, (self.batch_size, self.n), dtype=cp.float64)
        new_pop = cp.stack([random_thetas, random_rhos], axis=-1)
        self.generations_without_improvement = 0
        return new_pop
    
    def _get_hash(self, genome_cpu):
        arr = np.array(genome_cpu)
        ind = np.lexsort((arr[:, 1], arr[:, 0]))
        arr_sorted = arr[ind]
        arr_rounded = np.round(arr_sorted, 2)
        return hashlib.md5(arr_rounded.tobytes()).hexdigest()
    
    def _save_solution(self, genome_cpu, k_value, is_record=False):
        h = self._get_hash(genome_cpu)
        if not is_record and h in self.seen_hashes:
            return False
        
        self.seen_hashes.add(h)
        self.variant_counter += 1
        save_dir = "data/solutions_run2"
        os.makedirs(save_dir, exist_ok=True)
        
        filename = f"{save_dir}/N{self.n}_K{k_value}_{h[:8]}.json"
        lines_data = [{"id": i, "theta": float(genome_cpu[i, 0]), "rho": float(genome_cpu[i, 1])} for i in range(len(genome_cpu))]
        out = {"n": self.n, "k": k_value, "hash": h, "lines": lines_data}
        
        with open(filename, "w") as f:
            json.dump(out, f, indent=2)
        
        tag = "[!!!] NEW RECORD" if is_record else "[*] New Variant"
        print(f"{tag} K={k_value} -> {filename}")
        sys.stdout.flush()
        return True
    
    def step(self):
        # 1. Calculate raw fitness
        raw_fitness = calculate_fitness_batch(self.population)
        
        # 2. Apply Taboo Penalty (ANTI-GRAVITY)
        fitness, n_penalized = self._apply_taboo_penalty(raw_fitness)
        
        # 3. Evaluate
        current_max = int(cp.max(raw_fitness))  # Use RAW for record tracking
        best_idx = int(cp.argmax(raw_fitness))
        current_k = current_max
        
        # 4. Check for improvement
        if current_k > self.best_fitness:
            self.best_fitness = current_k
            self.best_genome = self.population[best_idx].copy()
            self.generations_without_improvement = 0
            self.consecutive_stagnations = 0
            
            genome_cpu = cp.asnumpy(self.best_genome)
            self.seen_hashes.clear()
            self._save_solution(genome_cpu, current_k, is_record=True)
        else:
            self.generations_without_improvement += 1
            
            if current_k >= 53:
                genome_cpu = cp.asnumpy(self.population[best_idx])
                saved = self._save_solution(genome_cpu, current_k, is_record=False)
                if saved:
                    self.generations_without_improvement = 0
                    self.consecutive_stagnations = 0
        
        # 5. Sort by PENALIZED fitness (taboo-aware selection)
        sorted_idx = cp.argsort(fitness)[::-1]
        sorted_pop = self.population[sorted_idx]
        
        # 6. Fast Fail
        self.fast_fail_counter += 1
        if self.fast_fail_counter > FAST_FAIL_LIMIT and self.best_fitness < FAST_FAIL_K:
            print(f"[FAST FAIL] K={self.best_fitness} after {self.fast_fail_counter} gens. Wiping.")
            self.population = self._trigger_hard_reset()
            self.fast_fail_counter = 0
            return current_k, n_penalized
        
        # 7. Stagnation
        if self.generations_without_improvement >= STAGNATION_LIMIT:
            self.consecutive_stagnations += 1
            print(f"[STAGNATION] #{self.consecutive_stagnations}. K={self.best_fitness}. HARD RESET.")
            self.population = self._trigger_hard_reset()
            return current_k, n_penalized
        
        # 8. Normal Evolution
        elites = sorted_pop[:self.elite_count]
        n_offspring = self.batch_size - self.elite_count
        
        top_half = self.batch_size // 2
        parents_idx = cp.random.randint(0, top_half, (n_offspring, 2))
        p1 = sorted_pop[parents_idx[:, 0]]
        p2 = sorted_pop[parents_idx[:, 1]]
        
        crossover_mask = cp.random.rand(n_offspring, self.n, 1) < 0.5
        offspring = cp.where(crossover_mask, p1, p2)
        
        mutate_th = cp.random.rand(n_offspring, self.n) < MUTATION_RATE
        noise_th = cp.random.normal(0, 0.05, (n_offspring, self.n))
        offspring[:, :, 0] += mutate_th * noise_th
        
        mutate_rho = cp.random.rand(n_offspring, self.n) < MUTATION_RATE
        noise_rho = cp.random.normal(0, 0.05, (n_offspring, self.n))
        offspring[:, :, 1] += mutate_rho * noise_rho
        
        offspring[:, :, 0] = offspring[:, :, 0] % cp.pi
        
        self.population = cp.concatenate([elites, offspring])
        return current_k, n_penalized

# ─── MAIN LOOP ──────────────────────────────────────────────────────────────

def main():
    # 1. Extract Taboo Core from hardcoded ALPHA_CORE_K53.json
    taboo_core = extract_taboo_core()
    
    # 2. Archive old solutions
    archive_solutions()
    
    # 3. Initialize
    opt = TabooOptimizer(n_lines=N_LINES, taboo_core=taboo_core, batch_size=BATCH_SIZE)
    
    print(f"\n{'='*60}")
    print(f"  TABOO SEARCH — Anti-Gravity Protocol")
    print(f"  Batch: {BATCH_SIZE} | Fast Fail: {FAST_FAIL_LIMIT}g/K<{FAST_FAIL_K}")
    print(f"  Stagnation: {STAGNATION_LIMIT}g | Target: K>={TARGET_K}")
    print(f"{'='*60}\n")
    
    gen = 0
    t0 = time.time()
    
    while True:
        gen += 1
        current_k, n_penalized = opt.step()
        
        if gen % 50 == 0:
            elapsed = time.time() - t0
            taboo_str = f" | Taboo Repulsions: {opt.total_taboo_hits}" if opt.total_taboo_hits > 0 else ""
            print(f"Gen {gen:>6} | Best K: {opt.best_fitness} | Current: {current_k} | Taboo Hits: {n_penalized}{taboo_str} | Time: {elapsed:.1f}s")
            sys.stdout.flush()
        
        if current_k >= TARGET_K:
            elapsed = time.time() - t0
            print(f"\n{'#'*60}")
            print(f"  !!! VICTORY !!! K={current_k} AT GEN {gen}")
            print(f"  Time: {elapsed:.2f}s")
            print(f"{'#'*60}\n")
            sys.exit(0)

if __name__ == "__main__":
    main()
