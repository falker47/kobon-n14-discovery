# Kobon Triangle Problem: The $K(14) = 54$ Discovery

![Python](https://img.shields.io/badge/Python-3.12-blue)
![CuPy](https://img.shields.io/badge/CuPy-GPU_Accelerated-green)
![Status](https://img.shields.io/badge/Status-Upper_Bound_Reached-success)

This repository documents the computational discovery of the exact $K=54$ configuration for the Kobon Triangle problem with $n=14$ lines, effectively closing the gap between the previously best-known empirical result ($K=53$) and the theoretical upper bound ($U=54$).

## The Configuration ($K=54$)

Below is the dynamic construction of the 14 lines and the strict validation of the 54 non-overlapping triangles.

![K=54 Construction](assets/HOLY_GRAIL_K54_CONSTRUCTION.gif)

## Abstract
The Kobon triangle problem asks for the maximum number of non-overlapping triangles $K(n)$ that can be formed by $n$ lines in the Euclidean plane. For $n=14$, the upper bound dictated by the Tamura equation (with Clément and Bader refinements) is 54. Until this project, the maximum empirically verified configuration stood at 53.

This project ("Project Hycarus") deployed a massively parallelized, GPU-accelerated Genetic Algorithm. The primary breakthrough was achieved by identifying a topological "Super-Attractor" (a sterile local minimum of $K=53$ that trapped standard stochastic search) and deploying a **Vectorial Taboo Search**. By mathematically penalizing this specific 6-line core skeleton, the algorithm was forced into an unexplored geometric basin, revealing the $K=54$ configuration.

## Repository Architecture

* `/data/solutions_run2/`: Contains `N14_K54_3c335303.json`, the raw coordinates of the winning configuration.
* `/data/solutions_archive.../`: The archival data of the 200+ K=53 variants forming the "Alpha Family" Super-Attractor.
* `/src/`: The Python source code.
    * `optimizer.py` / `main.py`: The core CuPy genetic engine.
    * `optimizer_taboo.py`: The Anti-Gravity protocol used to bypass the local minimum.
    * `verify_grail.py`: Strict CPU float64 validation logic.
* `/assets/`: High-resolution renders and animations.
* `/docs/`: Contains `hycarus_paper_draft.md`, the full academic manuscript detailing the methodology, variance analysis, and validation protocols.

## Verification
The coordinates provided in the JSON are verified via strict segment-intersection mathematics on the CPU (float64 precision), calculating intersections across all 364 possible line triplets to eliminate GPU floating-point hallucinations or optical aliasing.

To run the verification locally:
```bash
python src/verify_grail.py
```

## Future Work
The methodology of vector-space Taboo Search to ban topological Super-Attractors is currently being adapted to target higher unresolved dimensions, specifically $n=18$ and $n=20$.