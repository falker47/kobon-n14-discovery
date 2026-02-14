import cupy as cp
import numpy as np
import os
import json
import time
import hashlib
from geometry_engine import calculate_fitness_batch

class GeneticAlgorithm:
    def __init__(self, n_lines, batch_size=10000, mutation_rate=0.01, elite_ratio=0.05):
        self.n = n_lines
        self.batch_size = batch_size
        self.mutation_rate = mutation_rate
        self.elite_count = int(batch_size * elite_ratio)
        
        self.population = self._initialize_population()
        self.best_fitness = 0
        self.best_genome = None
        
        # STAGNATION BREAKER
        self.generations_without_improvement = 0
        self.stagnation_threshold = 2000
        self.elite_preserve_ratio = 0.01
        
        # SNIPER PROTOCOL
        self.sniper_mode_active = False
        self.sniper_k53_threshold = 53
        self.sniper_timeout = 5000
        self.k53_variant_counter = 0
        # COARSE DEDUPLICATION (2 decimal places)
        self.seen_coarse_hashes = set()
        
        # FAST FAIL COUNTER
        self.fast_fail_counter = 0
        
        # DYNAMIC EARTHQUAKE LOGIC
        self.consecutive_stagnations = 0

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

    def _trigger_earthquake(self, elites):
        
        n_elites = len(elites)
        
        # Force Reset Mode to EXPLORATION
        self.sniper_mode_active = False
        
        if n_elites == 0:
            print("[!!!] FULL WIPE: Generating 100% fresh random population.")
            # Full Nuke: No elites, so no mutants. All random.
            n_random = self.batch_size
            
            # Random Fresh Blood
            random_thetas = cp.random.uniform(0, cp.pi, (n_random, self.n), dtype=cp.float64)
            random_rhos = cp.random.uniform(-2.0, 2.0, (n_random, self.n), dtype=cp.float64)
            
            # EXPLICIT OVERWRITE
            new_population = cp.stack([random_thetas, random_rhos], axis=-1)
            
            self.generations_without_improvement = 0
            print(f"   [HARD RESET] Generated {n_random} completely random individuals.")
            return new_population

        print(f"[!] SOFT EARTHQUAKE: Preserving {n_elites} elites...")
        
        n_to_generate = self.batch_size - n_elites
        
        n_mutated = int(n_to_generate * 0.5)
        
        clone_indices = cp.random.randint(0, n_elites, n_mutated)
        mutated_clones = elites[clone_indices].copy()
        
        heavy_noise_th = cp.random.normal(0, 0.5, (n_mutated, self.n), dtype=cp.float64)
        heavy_noise_rho = cp.random.normal(0, 2.0, (n_mutated, self.n), dtype=cp.float64)
        mutated_clones[:, :, 0] += heavy_noise_th
        mutated_clones[:, :, 1] += heavy_noise_rho
        mutated_clones[:, :, 0] = mutated_clones[:, :, 0] % cp.pi
        
        n_random = n_to_generate - n_mutated
        
        # Random Fresh Blood
        random_thetas = cp.random.uniform(0, cp.pi, (n_random, self.n), dtype=cp.float64)
        random_rhos = cp.random.uniform(-2.0, 2.0, (n_random, self.n), dtype=cp.float64)
        random_pop = cp.stack([random_thetas, random_rhos], axis=-1)
        
        parts = [elites, mutated_clones, random_pop]
        new_population = cp.concatenate(parts)
        
        self.generations_without_improvement = 0
        print(f"   [SOFT RESET] Preserved {n_elites} elites, mutated {n_mutated}, random {n_random}.")
        
        return new_population

    def _apply_micro_surgery(self, sorted_pop):
        """
        SNIPER PROTOCOL: Micro-Surgery Mode.
        Apply extremely small perturbations to single lines in elite genomes.
        """
        elites = sorted_pop[:self.elite_count]
        n_offspring = self.batch_size - self.elite_count
        
        # Clone elites for offspring
        offspring_indices = cp.random.randint(0, self.elite_count, n_offspring)
        offspring = elites[offspring_indices].copy()
        
        # Micro-Surgery: very small sigma, targeting single lines
        # For each offspring, pick ONE random line to tweak
        line_to_tweak = cp.random.randint(0, self.n, n_offspring)
        
        # Create masks
        line_mask = cp.zeros((n_offspring, self.n), dtype=cp.float64)
        line_mask[cp.arange(n_offspring), line_to_tweak] = 1.0
        
        # Extremely small noise (sigma=1e-5)
        micro_noise_th = cp.random.normal(0, 1e-5, (n_offspring, self.n), dtype=cp.float64)
        micro_noise_rho = cp.random.normal(0, 1e-5, (n_offspring, self.n), dtype=cp.float64)
        
        offspring[:, :, 0] += line_mask * micro_noise_th
        offspring[:, :, 1] += line_mask * micro_noise_rho
        offspring[:, :, 0] = offspring[:, :, 0] % cp.pi
        
        return cp.concatenate([elites, offspring])

    def _get_coarse_hash(self, genome_cpu):
        """
        Compute a coarse hash (2 decimal places) for deduplication.
        Sorts lines to normalize permutations.
        """
        arr = np.array(genome_cpu)
        # Sort by theta, then rho
        ind = np.lexsort((arr[:, 1], arr[:, 0]))
        arr_sorted = arr[ind]
        # Round to 2 decimals (coarse)
        arr_rounded = np.round(arr_sorted, 2)
        return hashlib.md5(arr_rounded.tobytes()).hexdigest()

    def _save_solution(self, genome_cpu, k_value, is_new_record=False):
        """
        Save a solution with coarse deduplication.
        Returns True if saved, False if duplicate.
        """
        coarse_hash = self._get_coarse_hash(genome_cpu)
        
        # Check for duplicate (skip if already seen, unless new record)
        if not is_new_record and coarse_hash in self.seen_coarse_hashes:
            return False
        
        self.seen_coarse_hashes.add(coarse_hash)
        self.k53_variant_counter += 1
        
        os.makedirs("solutions", exist_ok=True)
        
        # Use hash suffix for filename
        hash_suffix = coarse_hash[:8]
        filename = f"solutions/N{self.n}_K{k_value}_{hash_suffix}.json"
        
        lines_data = []
        for i in range(len(genome_cpu)):
            lines_data.append({
                "id": i,
                "theta": float(genome_cpu[i, 0]),
                "rho": float(genome_cpu[i, 1])
            })
        
        out_data = {"n": self.n, "k": k_value, "hash": coarse_hash, "lines": lines_data}
        
        with open(filename, "w") as f:
            json.dump(out_data, f, indent=2)
        
        if is_new_record:
            print("[!!!] NEW RECORD K={} saved -> {}".format(k_value, filename))
        else:
            print("[*] New Distinct Base Camp K={} -> {}".format(k_value, filename))
        return True

    def step(self):
        fitness = calculate_fitness_batch(self.population)
        
        current_max = cp.max(fitness)
        best_idx = cp.argmax(fitness)
        
        improved = False
        current_k = int(current_max)
        
        # Case A: New World Record
        if current_k > self.best_fitness:
            self.best_fitness = current_k
            self.best_genome = self.population[best_idx].copy()
            improved = True
            self.generations_without_improvement = 0
            self.generations_without_improvement = 0
            self.consecutive_stagnations = 0 # Reset stagnation counter on improvement
            
            # ALWAYS save new records, clear hash cache
            genome_cpu = cp.asnumpy(self.best_genome)
            self.seen_coarse_hashes.clear()
            self._save_solution(genome_cpu, current_k, is_new_record=True)
            
        else:
            self.generations_without_improvement += 1
            
            # Case B: Tied at K>=53 - check for distinct base camp
            if current_k >= 53:
                genome_cpu = cp.asnumpy(self.population[best_idx])
                saved = self._save_solution(genome_cpu, current_k, is_new_record=False)
                if saved:
                     self.generations_without_improvement = 0  # Treat finding a new variant as improvement
                     self.consecutive_stagnations = 0
        
        sorted_indices = cp.argsort(fitness)[::-1]
        sorted_pop = self.population[sorted_indices]
        
        # SNIPER PROTOCOL: Enter Micro-Surgery if CURRENT population is at K=53
        # Logic: We only want to micro-surgery if we are actually ON the peak with this population.
        # "Zombie Sniper" Fix: Check current_k, not self.best_fitness
        if current_k >= self.sniper_k53_threshold and current_k < 54:
            if not self.sniper_mode_active:
                print(f"[*] SNIPER MODE ACTIVATED: Entering Micro-Surgery (Current K={current_k})...")
                self.sniper_mode_active = True
            
            # Check sniper timeout
            if self.generations_without_improvement >= self.sniper_timeout:
                self.consecutive_stagnations += 1
                # HYPER-EXPLORATION: ALWAYS HARD RESET
                print(f"[TIMEOUT] Sniper timeout. Stagnation #{self.consecutive_stagnations}. HARD RESET (0 elites).")
                empty_elites = sorted_pop[:0]
                self.population = self._trigger_earthquake(empty_elites)
                
                return current_k
            
            # Apply micro-surgery
            self.population = self._apply_micro_surgery(sorted_pop)
            return current_k
        else:
             # Ensure we exit sniper mode if we drop below K=53 (e.g. after earthquake)
             if self.sniper_mode_active:
                  self.sniper_mode_active = False
                  print("[*] SNIPER MODE DEACTIVATED: Dropped below K=53")
        
        # FAST FAIL CHECK (TTL) — HYPER-EXPLORATION: 1000 gens, K<52
        self.fast_fail_counter += 1
        if self.fast_fail_counter > 1000 and self.best_fitness < 52:
             print(f"FAST FAIL TRIGGERED: Performance too low (K={self.best_fitness} after {self.fast_fail_counter} gens). Wiping population.")
             # Hard Reset
             empty_elites = sorted_pop[:0]
             self.population = self._trigger_earthquake(empty_elites)
             self.fast_fail_counter = 0
             return current_k

        # Normal mode: Check stagnation
        # Aggressive limit: 1500 generations (for Stuck at K=52 etc, if Fast Fail didn't catch it)
        if self.generations_without_improvement >= 1500:
            print(f"[STAGNATION] Stuck at K={self.best_fitness} for {self.generations_without_improvement} gens.")
            
            if self.best_fitness < 53:
                 print("[!!!] LOW-LEVEL STAGNATION (K<53). FULL RESET.")
                 # Nuke all - no elites. Force fresh search.
                 empty_elites = sorted_pop[:0] 
                 self.population = self._trigger_earthquake(empty_elites)
            else:
                 # HYPER-EXPLORATION: ALWAYS HARD RESET (even at K>=53)
                 self.consecutive_stagnations += 1
                 print(f"[STAGNATION] Cycle #{self.consecutive_stagnations}. HARD RESET (0 elites). Forcing new topology.")
                 empty_elites = sorted_pop[:0]
                 self.population = self._trigger_earthquake(empty_elites)
            
            return current_k
        
        # Normal evolution
        elites = sorted_pop[:self.elite_count]
        n_offspring = self.batch_size - self.elite_count
        
        top_half = self.batch_size // 2
        parents_indices = cp.random.randint(0, top_half, (n_offspring, 2))
        
        p1 = sorted_pop[parents_indices[:, 0]]
        p2 = sorted_pop[parents_indices[:, 1]]
        
        crossover_mask = cp.random.rand(n_offspring, self.n, 1) < 0.5
        offspring = cp.where(crossover_mask, p1, p2)
        
        mutate_mask_th = cp.random.rand(n_offspring, self.n) < self.mutation_rate
        noise_th = cp.random.normal(0, 0.05, (n_offspring, self.n))
        offspring[:, :, 0] += mutate_mask_th * noise_th
        
        mutate_mask_rho = cp.random.rand(n_offspring, self.n) < self.mutation_rate
        noise_rho = cp.random.normal(0, 0.05, (n_offspring, self.n))
        offspring[:, :, 1] += mutate_mask_rho * noise_rho
        
        offspring[:, :, 0] = offspring[:, :, 0] % cp.pi
        
        self.population = cp.concatenate([elites, offspring])
        
        return current_k
        
    def get_best_genome_cpu(self):
        if self.best_genome is None:
            return None
        return cp.asnumpy(self.best_genome)
