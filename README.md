# Partcl/HRT Macro Placement Challenge
# SpectralSAPlacer

**Spectral FFT Packing + Simulated Annealing macro placer**
Partcl/HRT Macro Placement Challenge 2026

## Author
Pushkar Ambastha (GitHub: [Pushkar1853](https://github.com/Pushkar1853))

## Approach

This placer combines two ideas: spectral packing from computational geometry
and connectivity-aware simulated annealing from classical EDA.

**Step 1 — Spectral FFT Placement** (inspired by MIT SIGGRAPH 2023 paper
*"Dense, Interlocking-Free and Scalable Spectral Packing"*):
- Represent the occupied canvas as a 2D occupancy grid
- For each macro, compute FFT cross-correlation with the occupancy map to find
  all collision-free positions in O(N log N) — instead of O(N²) brute force
- Among valid positions, pick the one minimizing weighted distance to
  already-placed connected macros (proximity metric, also via FFT)
- Place macros largest-first for best packing density

**Step 2 — Two-phase Will SA Refinement:**
- Connectivity-aware simulated annealing with four move types:
  shift, swap (neighbor-biased), move-toward-neighbor, rotate
- O(N) per-move overlap check using separation matrices
- Two phases: broad exploration then fine-tuning

**Step 3 — Legalization + Corner-based Overlap Resolution:**
- Will's minimum-displacement legalization (largest macros first)
- Final corner-based overlap resolver that matches the evaluator's
  exact overlap definition — guarantees zero overlaps on submission

## Why Spectral Initialization Helps

Classical SA and GNN approaches start from a random or learned initialization
that often has many overlaps, forcing SA to waste iterations just removing
collisions. The FFT approach gives SA a nearly overlap-free starting point
where macros are already roughly clustered by connectivity — SA can then
focus entirely on wirelength optimization.

## Files
- `submissions/pushkarambastha/placer.py` — complete placer implementation (no weights needed)

## No Training Required
This is a purely algorithmic approach. There are no neural network weights
to load — the placer runs end-to-end from scratch on any benchmark in under
the 1-hour time limit.

## Runtime
~3-6 minutes per benchmark on CPU. Runs well within the 1-hour limit.

## Dependencies
- `torch`, `numpy`, `scipy` (FFT)
- `torch-geometric` (graph data structures)
- All available in the competition evaluation environment

