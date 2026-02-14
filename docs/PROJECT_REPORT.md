# Project Report: Computational Search for Kobon Triangle Bounds
**Codename:** Hycarus / Kobon-14
**Date:** February 2026
**Author:** Falker47
**Hardware Context:** AMD Ryzen 7 5800H | NVIDIA RTX 3060 Laptop (6GB VRAM)

---

## 1. Executive Summary
This project addresses the **Kobon Triangle Problem**, an open challenge in combinatorial geometry asking for the maximum number of non-overlapping triangles ($K$) formed by $N$ lines.
Using a custom Python engine accelerated via **CUDA (CuPy)**, the project successfully:
1.  **Replicated the Optimal Solution for N=10** ($K=25$) in under 60 seconds (Calibration Phase).
2.  **Tied the World Record for N=14** ($K=53$) in approx. 180 seconds.
3.  **Identified and Fixed** critical floating-point precision errors inherent to geometric optimization.
4.  Currently executing a **"Fast Fail" Evolutionary Strategy** to hunt for the theoretical maximum $K=54$.

---

## 2. Mathematical Framework & Constraints

### The Problem
Given $N$ lines in the Euclidean plane, maximize $K(N)$ such that no two triangles overlap.
* **Tamura’s Upper Bound:** $K \le \lfloor N(N-2)/3 \rfloor$
* **Clement & Bader Improved Bound (N=14):** $K \le 54$
* **Current Best Known Solution (N=14):** $K = 53$ (Tied by this project)

### The Search Space
For $N=14$, the configuration space is 28-dimensional (continuous).
* **Representation:** Hessian Normal Form $(\rho, \theta)$ was chosen over slope-intercept $(m, q)$ to avoid singularities with vertical lines ($\theta = \pi/2$).
* **Topological Challenge:** The difference between $K=53$ and $K=54$ often lies in microscopic shifts of line clusters, requiring `float64` (double precision) accuracy.

---

## 3. Technical Architecture

The engine maximizes the **Hycarus** hardware (RTX 3060) by offloading linear algebra to the GPU while using the CPU for orchestration.

| Component | Technology | Implementation Details |
| :--- | :--- | :--- |
| **Math Kernel** | `CuPy` (CUDA) | Vectorized calculation of intersection points $(x, y)$ for batch sizes of 15,000 configurations. Solves linear systems via Cramer's Rule in parallel. |
| **Geometry Engine** | `NumPy` / Custom | **Strict Segment Intersection Logic**: Replaced "Signed Distance" checks (susceptible to floating-point drift) with rigorous edge-intersection tests to eliminate false positives. |
| **Optimizer** | Evolutionary Strategy | Implements "Earthquake" (Population Reshuffling) and "Sniper Mode" (Micro-variance $\sigma=10^{-5}$) to escape local minima. |
| **I/O Handler** | `Hashlib` | **In-Memory Deduplication**: Uses coarse hashing (2-decimal rounding) to identify and save only topologically distinct "Base Camps," preventing disk flooding. |

---

## 4. Algorithmic Evolution & Milestones

### Phase 1: Calibration (N=10)
* **Goal:** Verify engine logic against a solved problem ($Max K=25$).
* **Result:** Algorithm converged to $K=25$ consistently. Validated the "Perturbed Regular Polygon" initialization strategy.
* **Visual Proof:**
    ![N=10 K=25 Solution](path/to/solution_n10_k25_verified.jpg)

### Phase 2: The "False Summit" (Debugging)
* **Issue:** The engine reported a $K=54$ solution. Forensic analysis revealed "Phantom Triangles"—lines passing $10^{-5}$ units from a vertex were counted as valid due to loose tolerance (`1e-4`).
* **Fix:**
    1.  Tightened tolerance to `1e-9`.
    2.  Switched validation logic from Area-based to **Segment-based** (a triangle is valid IFF no line cuts its edges).
* **Outcome:** False positive rejected. True performance stabilized at $K=51$.

### Phase 3: The Earthquake Protocol (World Record Tie)
* **Challenge:** Stagnation at Local Minima ($K=51$).
* **Solution:** Implemented **Adaptive Restart**.
    * *Trigger:* 2000 generations without improvement.
    * *Action:* Keep top 1% Elites, randomize 99% of population.
* **Breakthrough:** On Gen 195 (post-reset), the system found a valid **$K=53$**.
* **Visual Proof:**
    ![N=14 K=53 Solution](path/to/solution_N14_K53.jpg)

### Phase 4: Current Strategy (Fast Fail & Sniper V2)
* **Hypothesis:** Finding $K=54$ is a "Lucky Seed" problem. Long optimization runs on bad seeds are inefficient.
* **Protocol:**
    * If $K < 53$ after 2000 generations $\rightarrow$ **FAST FAIL** (Wipe population / Hard Reset).
    * **Zombie Sniper Fix:** Corrected logic where `Earthquake` events could trigger "Micro-Surgery" mode on fresh, low-quality populations. The system now enforces `Exploration Mode` (High Variance) immediately after a reset.
    * Focus computational power on rapid cycling of new topological families.

---

## 5. Results Table

| Parameter | Value | Notes |
| :--- | :--- | :--- |
| **Max Triangles (N=10)** | **25** | Optimal (Proven) |
| **Max Triangles (N=14)** | **53** | **Tied World Record** |
| **Time to Record (N=14)** | ~180 sec | GPU-Accelerated |
| **Precision** | `float64` | Necessary for N=14 |
| **Unique Solutions Found** | >150 | Distinct topological "species" of K=53 |

---

## 6. Future Development
* **Distributed Search:** Deploy the "Fast Fail" container to a cloud cluster to increase seed throughput.
* **Topological Analysis:** Use graph theory to analyze the 150+ unique $K=53$ solutions found, looking for common sub-structures that could hint at the $K=54$ configuration.