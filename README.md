# Partcl/HRT Macro Placement Challenge

# GNNSAPlacer
GNN + Simulated Annealing macro placer for the Partcl/HRT Macro Placement Challenge.

## Author
Pushkar Ambastha (GitHub: Pushkar1853)

## Approach
- GraphSAGE GNN encoder learns macro embeddings from netlist connectivity
- RL policy (REINFORCE) places macros sequentially using learned embeddings
- Will's SA refinement polishes the GNN placement with connectivity-aware moves
- Final legalization guarantees zero overlaps

## Files
- `placer.py` — placer implementation
- `weights.pt` — trained GNN+policy weights
