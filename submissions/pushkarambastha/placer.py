# submissions/pushkarambastha/placer.py
import os, sys, subprocess, torch, numpy as np, random, math
from pathlib import Path

# Auto-install torch_geometric if missing
try:
    from torch_geometric.nn import SAGEConv
except ImportError:
    import subprocess, sys
    # Use uv (available in the competition environment)
    subprocess.run(
        ['uv', 'add', 'torch-geometric',
         '--find-links', 'https://data.pyg.org/whl/torch-2.3.0+cu121.html'],
        check=False  # don't crash if uv also fails
    )
    # Also try system pip as fallback
    subprocess.run(
        ['/usr/bin/pip3', 'install', '-q', 'torch-geometric',
         '--find-links', 'https://data.pyg.org/whl/torch-2.3.0+cu121.html'],
        check=False
    )
    from torch_geometric.nn import SAGEConv

from macro_place.benchmark import Benchmark
import torch.nn as nn
import torch.nn.functional as F

# ---- paste GNNEncoder class here ----
class GNNEncoder(nn.Module):
    def __init__(self, in_dim=6, hidden_dim=128, num_layers=3, dropout=0.1):
        super().__init__()
        self.convs = nn.ModuleList()
        self.bns   = nn.ModuleList()
        dims = [in_dim] + [hidden_dim] * num_layers
        for i in range(num_layers):
            self.convs.append(SAGEConv(dims[i], dims[i+1]))
            self.bns.append(nn.BatchNorm1d(dims[i+1]))
        self.dropout = dropout
        self.out_dim = hidden_dim

    def forward(self, x, edge_index):
        for conv, bn in zip(self.convs, self.bns):
            x = conv(x, edge_index)
            x = bn(x)
            x = F.relu(x)
            x = F.dropout(x, p=self.dropout, training=self.training)
        return x

# ---- paste PlacementPolicy class here ----
GRID       = 16
N_ROT      = 2
ACTION_DIM = GRID * GRID * N_ROT
# print(f'Grid: {GRID}x{GRID}, Action space: {ACTION_DIM}')
class PlacementPolicy(nn.Module):
    def __init__(self, emb_dim=128, canvas_dim=GRID*GRID, hidden=256):
        super().__init__()
        in_dim = emb_dim + canvas_dim
        self.shared = nn.Sequential(
            nn.Linear(in_dim, hidden), nn.ReLU(),
            nn.Linear(hidden, hidden), nn.ReLU(),
        )
        self.policy_head = nn.Linear(hidden, ACTION_DIM)
        self.value_head  = nn.Linear(hidden, 1)

    def forward(self, macro_emb, canvas_grid):
        # macro_emb:   [B, 128]
        # canvas_grid: [B, GRID, GRID]
        grid_flat = canvas_grid.view(canvas_grid.size(0), -1).float()
        x = torch.cat([macro_emb, grid_flat], dim=-1)
        h = self.shared(x)
        return self.policy_head(h), self.value_head(h)

# ---- paste PlacementEnv class here (with will_sa_refine) ----
from dataclasses import dataclass
import copy, math, random

@dataclass
class Position:
    x: float
    y: float
    rotation: int

class PlacementEnv:
    def __init__(self, bm, plc, evaluator_fn, grid_size=GRID):
        self.bm        = bm
        self.plc       = plc
        self.evaluator = evaluator_fn
        self.G         = grid_size
        self.W         = bm.canvas_width
        self.H         = bm.canvas_height
        self.cell_w    = self.W / grid_size
        self.cell_h    = self.H / grid_size
        num_hard       = bm.num_hard_macros
        sizes          = bm.macro_sizes[:num_hard]
        areas          = sizes[:, 0] * sizes[:, 1]
        self.order     = torch.argsort(areas, descending=True).tolist()
        self.reset()

    def reset(self):
        self.canvas           = np.zeros((self.G, self.G), dtype=np.float32)
        self.steps_done       = 0
        self.placed_positions = self.bm.macro_positions[:self.bm.num_hard_macros].clone()

    def canvas_tensor(self):
        return torch.tensor(self.canvas, dtype=torch.float32).unsqueeze(0)

    def _action_to_pos(self, action, macro_idx):
        rot   = action % N_ROT
        cell  = action // N_ROT
        row   = cell // self.G
        col   = cell  % self.G
        sizes = self.bm.macro_sizes[macro_idx]
        w     = sizes[1].item() if rot == 1 else sizes[0].item()
        h     = sizes[0].item() if rot == 1 else sizes[1].item()
        cx    = col * self.cell_w + self.cell_w / 2
        cy    = row * self.cell_h + self.cell_h / 2
        x     = min(max(cx - w / 2, 0), self.W - w)
        y     = min(max(cy - h / 2, 0), self.H - h)
        return Position(x=x, y=y, rotation=rot * 90)

    def _update_canvas(self, macro_idx, pos):
        sizes = self.bm.macro_sizes[macro_idx]
        w     = sizes[1].item() if pos.rotation == 90 else sizes[0].item()
        h     = sizes[0].item() if pos.rotation == 90 else sizes[1].item()
        c0    = int(pos.x / self.cell_w)
        c1    = min(int((pos.x + w) / self.cell_w) + 1, self.G)
        r0    = int(pos.y / self.cell_h)
        r1    = min(int((pos.y + h) / self.cell_h) + 1, self.G)
        self.canvas[r0:r1, c0:c1] = 1.0

    def get_invalid_mask(self, macro_idx):
        mask  = np.zeros(ACTION_DIM, dtype=bool)
        sizes = self.bm.macro_sizes[macro_idx]
        for rot in range(N_ROT):
            w  = sizes[1].item() if rot == 1 else sizes[0].item()
            h  = sizes[0].item() if rot == 1 else sizes[1].item()
            cw = int(np.ceil(w / self.cell_w))
            ch = int(np.ceil(h / self.cell_h))
            for row in range(self.G):
                for col in range(self.G):
                    oob  = (col + cw > self.G) or (row + ch > self.G)
                    occ  = self.canvas[row:row+ch, col:col+cw].sum() > 0
                    mask[(row * self.G + col) * N_ROT + rot] = oob or occ
        return mask

    def step(self, macro_idx, action):
        pos = self._action_to_pos(action, macro_idx)
        self.placed_positions[macro_idx] = torch.tensor([pos.x, pos.y])
        self._update_canvas(macro_idx, pos)
        self.steps_done += 1
        done   = (self.steps_done == len(self.order))
        reward = 0.0
        if done:
            reward = -self.evaluator(self.bm, self.plc, self.placed_positions)
        return pos

    def will_sa_refine(self, gnn_positions: torch.Tensor,
                       graph,
                       n_iters: int = 8000,
                       verbose: bool = False) -> torch.Tensor:
        """
        Takes GNN-produced positions [num_hard, 2] and refines with Will's SA.
        Uses edge_weight from graph for connectivity-aware moves.
        Returns improved [num_hard, 2] tensor.
        """
        bm       = self.bm
        n_hard   = bm.num_hard_macros
        W, H     = bm.canvas_width, bm.canvas_height
        sizes_np = bm.macro_sizes[:n_hard].numpy().astype(np.float64)
        half_w   = sizes_np[:, 0] / 2
        half_h   = sizes_np[:, 1] / 2

        # movable mask — fixed macros don't move
        if hasattr(bm, 'get_movable_mask'):
            movable = bm.get_movable_mask()[:n_hard].numpy()
        else:
            movable = ~bm.macro_fixed[:n_hard].numpy()

        movable_idx = np.where(movable)[0]
        if len(movable_idx) == 0:
            return gnn_positions

        # Build separation matrices
        sep_x = (sizes_np[:, 0:1] + sizes_np[:, 0:1].T) / 2
        sep_y = (sizes_np[:, 1:2] + sizes_np[:, 1:2].T) / 2

        # Build neighbor lists from graph edges (hard macros only)
        neighbors = [[] for _ in range(n_hard)]
        ei = graph.edge_index.cpu().numpy()
        ew = graph.edge_weight.cpu().numpy() if hasattr(graph, 'edge_weight') and graph.edge_weight is not None else np.ones(ei.shape[1])
        for k in range(ei.shape[1]):
            i, j = int(ei[0, k]), int(ei[1, k])
            if i < n_hard and j < n_hard:
                neighbors[i].append((j, ew[k]))
                # undirected already in edge_index

        pos = gnn_positions.numpy().astype(np.float64).copy()

        # Step 1: legalize GNN output first
        pos = self._legalize(pos, movable, sizes_np, half_w, half_h, W, H, n_hard)

        def wl_cost(p):
            if ei.shape[1] == 0:
                return 0.0
            # Ensure both source and destination nodes are hard macros
            hard_macro_edge_mask = (ei[0] < n_hard) & (ei[1] < n_hard)
            idx_i = ei[0][hard_macro_edge_mask]
            idx_j = ei[1][hard_macro_edge_mask]
            w_sel = ew[hard_macro_edge_mask]
            dx    = np.abs(p[idx_i, 0] - p[idx_j, 0])
            dy    = np.abs(p[idx_i, 1] - p[idx_j, 1])
            return (w_sel * (dx + dy)).sum()

        def check_overlap(idx, p):
            gap = 0.00
            dx  = np.abs(p[idx, 0] - p[:, 0])
            dy  = np.abs(p[idx, 1] - p[:, 1])
            ov  = (dx < sep_x[idx] + gap) & (dy < sep_y[idx] + gap)
            ov[idx] = False
            return ov.any()

        current_cost = wl_cost(pos)
        best_pos     = pos.copy()
        best_cost    = current_cost

        T_start = max(W, H) * 0.15
        T_end   = max(W, H) * 0.001

        for step in range(n_iters):
            frac = step / n_iters
            T    = T_start * (T_end / T_start) ** frac
            move = random.random()
            i    = random.choice(movable_idx)
            ox, oy = pos[i, 0], pos[i, 1]

            if move < 0.45:
                # SHIFT
                shift    = T * (0.3 + 0.7 * (1 - frac))
                pos[i,0] = np.clip(pos[i,0] + random.gauss(0, shift), half_w[i], W - half_w[i])
                pos[i,1] = np.clip(pos[i,1] + random.gauss(0, shift), half_h[i], H - half_h[i])

            elif move < 0.70:
                # SWAP — prefer connected neighbors
                nb_list = [j for j, _ in neighbors[i] if movable[j]] if neighbors[i] else []
                j = random.choice(nb_list) if nb_list and random.random() < 0.7 else random.choice(movable_idx)
                if i == j:
                    continue
                ojx, ojy = pos[j, 0], pos[j, 1]
                pos[i,0] = np.clip(ojx, half_w[i], W - half_w[i])
                pos[i,1] = np.clip(ojy, half_h[i], H - half_h[i])
                pos[j,0] = np.clip(ox,  half_w[j], W - half_w[j])
                pos[j,1] = np.clip(oy,  half_h[j], H - half_h[j])
                if check_overlap(i, pos) or check_overlap(j, pos):
                    pos[i,0]=ox; pos[i,1]=oy; pos[j,0]=ojx; pos[j,1]=ojy
                    continue
                new_cost = wl_cost(pos)
                delta    = new_cost - current_cost
                if delta < 0 or random.random() < math.exp(-delta / max(T, 1e-10)):
                    current_cost = new_cost
                    if current_cost < best_cost:
                        best_cost = current_cost; best_pos = pos.copy()
                else:
                    pos[i,0]=ox; pos[i,1]=oy; pos[j,0]=ojx; pos[j,1]=ojy
                continue

            elif move < 0.85:
                # MOVE TOWARD NEIGHBOR
                if neighbors[i]:
                    j, _     = random.choice(neighbors[i])
                    alpha    = random.uniform(0.05, 0.3)
                    pos[i,0] = np.clip(pos[i,0]+alpha*(pos[j,0]-pos[i,0]), half_w[i], W-half_w[i])
                    pos[i,1] = np.clip(pos[i,1]+alpha*(pos[j,1]-pos[i,1]), half_h[i], H-half_h[i])
                else:
                    continue

            else:
                # ROTATE — swap width/height, re-clamp
                pos[i,0] = np.clip(pos[i,0], half_h[i], W - half_h[i])
                pos[i,1] = np.clip(pos[i,1], half_w[i], H - half_h[i])
                # update sep_x/sep_y for this macro
                sizes_np[i, 0], sizes_np[i, 1] = sizes_np[i, 1], sizes_np[i, 0]
                half_w[i], half_h[i] = half_h[i], half_w[i]
                sep_x = (sizes_np[:, 0:1] + sizes_np[:, 0:1].T) / 2
                sep_y = (sizes_np[:, 1:2] + sizes_np[:, 1:2].T) / 2

            # Single-macro overlap check O(N)
            if check_overlap(i, pos):
                pos[i,0] = ox; pos[i,1] = oy
                # undo rotate if that was the move
                if move >= 0.85:
                    sizes_np[i,0], sizes_np[i,1] = sizes_np[i,1], sizes_np[i,0]
                    half_w[i], half_h[i] = half_h[i], half_w[i]
                    sep_x = (sizes_np[:,0:1] + sizes_np[:,0:1].T) / 2
                    sep_y = (sizes_np[:,1:2] + sizes_np[:,1:2].T) / 2
                continue

            new_cost = wl_cost(pos)
            delta    = new_cost - current_cost
            if delta < 0 or random.random() < math.exp(-delta / max(T, 1e-10)):
                current_cost = new_cost
                if current_cost < best_cost:
                    best_cost = current_cost; best_pos = pos.copy()
            else:
                pos[i,0] = ox; pos[i,1] = oy
                if move >= 0.85:
                    sizes_np[i,0], sizes_np[i,1] = sizes_np[i,1], sizes_np[i,0]
                    half_w[i], half_h[i] = half_h[i], half_w[i]
                    sep_x = (sizes_np[:,0:1] + sizes_np[:,0:1].T) / 2
                    sep_y = (sizes_np[:,1:2] + sizes_np[:,1:2].T) / 2

            # if verbose and step % 1000 == 0:
            #     print(f'  SA step {step:5d} | T={T:.3f} | cost={current_cost:.4f} | best={best_cost:.4f}')

        best_pos = self._legalize(best_pos, movable, sizes_np, half_w, half_h, W, H, n_hard)
        return torch.tensor(best_pos, dtype=torch.float32)

    def _legalize(self, pos, movable, sizes, half_w, half_h, W, H, n):
        """Will's legalization — minimum displacement, largest first."""
        sep_x  = (sizes[:, 0:1] + sizes[:, 0:1].T) / 2
        sep_y  = (sizes[:, 1:2] + sizes[:, 1:2].T) / 2
        order  = sorted(range(n), key=lambda i: -sizes[i,0] * sizes[i,1])
        placed = np.zeros(n, dtype=bool)
        legal  = pos.copy()

        for idx in order:
            if not movable[idx]:
                placed[idx] = True; continue
            if placed.any():
                dx = np.abs(legal[idx,0] - legal[:,0])
                dy = np.abs(legal[idx,1] - legal[:,1])
                c  = (dx < sep_x[idx]+0.05) & (dy < sep_y[idx]+0.05) & placed
                c[idx] = False
                if not c.any():
                    placed[idx] = True; continue

            step   = max(sizes[idx,0], sizes[idx,1]) * 0.25
            best_p = legal[idx].copy(); best_d = float('inf')

            for r in range(1, 150):
                found = False
                for dxm in range(-r, r+1):
                    for dym in range(-r, r+1):
                        if abs(dxm) != r and abs(dym) != r: continue
                        cx = np.clip(pos[idx,0]+dxm*step, half_w[idx], W-half_w[idx])
                        cy = np.clip(pos[idx,1]+dym*step, half_h[idx], H-half_h[idx])
                        if placed.any():
                            dx = np.abs(cx-legal[:,0]); dy = np.abs(cy-legal[:,1])
                            c  = (dx < sep_x[idx]+0.05) & (dy < sep_y[idx]+0.05) & placed
                            c[idx] = False
                            if c.any(): continue
                        d = (cx-pos[idx,0])**2 + (cy-pos[idx,1])**2
                        if d < best_d:
                            best_d = d; best_p = np.array([cx, cy]); found = True
                if found: break
            legal[idx] = best_p; placed[idx] = True

        return legal

    def will_sa_refine_2phase(self, gnn_positions, graph, verbose=False):
        phase1 = self.will_sa_refine(gnn_positions, graph, n_iters=5000, verbose=verbose)
        phase2 = self.will_sa_refine(phase1,        graph, n_iters=5000, verbose=verbose)
        # Extra legalization at the very end
        bm       = self.bm
        n_hard   = bm.num_hard_macros
        W, H     = bm.canvas_width, bm.canvas_height
        sizes_np = bm.macro_sizes[:n_hard].numpy().astype(np.float64)
        half_w   = sizes_np[:, 0] / 2
        half_h   = sizes_np[:, 1] / 2
        if hasattr(bm, 'get_movable_mask'):
            movable = bm.get_movable_mask()[:n_hard].numpy()
        else:
            movable = ~bm.macro_fixed[:n_hard].numpy()
        pos = phase2.numpy().astype(np.float64)
        pos = self._legalize(pos, movable, sizes_np, half_w, half_h, W, H, n_hard)
        return torch.tensor(pos, dtype=torch.float32)

# ---- paste benchmark_to_graph here ----
import torch
import numpy as np
from torch_geometric.data import Data

def benchmark_to_graph(bm, plc) -> Data:
    W, H     = bm.canvas_width, bm.canvas_height
    N        = bm.num_macros
    num_hard = bm.num_hard_macros
    sizes    = bm.macro_sizes
    pos      = bm.macro_positions
    fixed    = bm.macro_fixed.float().unsqueeze(1)

    feats = torch.cat([
        sizes[:, 0:1] / W,
        sizes[:, 1:2] / H,
        (sizes[:, 0:1] * sizes[:, 1:2]) / (W * H),
        fixed,
        pos[:, 0:1] / W,
        pos[:, 1:2] / H,
    ], dim=1)

    # Build edges from plc.nets with weights (like Will's _extract_edges)
    src_list, dst_list, weight_list = [], [], []
    try:
        all_names         = bm.macro_names
        name_to_graph_idx = {name: i for i, name in enumerate(all_names)}
        edge_dict         = {}

        for driver, sinks in plc.nets.items():
            members = set()
            for pin in [driver] + sinks:
                macro_name = pin.split('/')[0]
                if macro_name in name_to_graph_idx:
                    members.add(name_to_graph_idx[macro_name])
            members = sorted(members)
            if len(members) < 2:
                continue
            w = 1.0 / (len(members) - 1)  # weight inversely prop to net size
            for i in range(len(members)):
                for j in range(i + 1, len(members)):
                    pair = (members[i], members[j])
                    edge_dict[pair] = edge_dict.get(pair, 0) + w

        for (i, j), w in edge_dict.items():
            src_list += [i, j]
            dst_list += [j, i]
            weight_list += [w, w]

        # print(f'Built {len(edge_dict)} edges from {len(plc.nets)} nets')

    except Exception as e:
        print(f'Edge building failed: {e}')

    if src_list:
        edge_index  = torch.tensor([src_list, dst_list], dtype=torch.long)
        edge_weight = torch.tensor(weight_list, dtype=torch.float32)
    else:
        edge_index  = torch.zeros((2, 0), dtype=torch.long)
        edge_weight = torch.zeros(0, dtype=torch.float32)

    data            = Data(x=feats, edge_index=edge_index, edge_weight=edge_weight)
    data.num_hard   = num_hard
    data.meta       = {'W': W, 'H': H, 'N': N}
    return data

# ---- paste evaluator_fn here ----
from macro_place.objective import compute_proxy_cost
from macro_place.utils import validate_placement

def evaluator_fn(bm, plc, hard_positions: torch.Tensor):
    """
    hard_positions: [num_hard, 2] tensor — only hard macro positions
    Evaluator needs all macros, so we concatenate soft macro positions from bm.
    """
    soft_positions = bm.macro_positions[bm.num_hard_macros:]  # [num_soft, 2]
    all_positions  = torch.cat([hard_positions, soft_positions], dim=0)  # [N, 2]
    costs = compute_proxy_cost(all_positions, bm, plc)
    return costs['proxy_cost']

# Smoke test
# hard_pos = bm.macro_positions[:bm.num_hard_macros]
# cost = evaluator_fn(bm, plc, hard_pos)
# print(f'Proxy cost (initial placement): {cost:.4f}')

# Also check validate_placement
# is_valid, violations = validate_placement(bm.macro_positions, bm)
# print(f'Initial placement valid: {is_valid}')
# print(f'Violations: {len(violations)}')

# ---- paste run_episode here ----
from torch.distributions import Categorical
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
def run_episode(bm, plc, graph, encoder, policy, greedy=False):
    """
    Returns: log_probs, values, reward (scalar), positions [num_hard, 2]
    """
    W, H     = bm.canvas_width, bm.canvas_height
    num_hard = bm.num_hard_macros
    sizes    = bm.macro_sizes[:num_hard]  # [num_hard, 2]

    # Sort order: largest area first
    areas   = sizes[:, 0] * sizes[:, 1]
    order   = torch.argsort(areas, descending=True).tolist()

    env     = PlacementEnv(bm, plc, evaluator_fn, grid_size=GRID)
    env.reset()

    embeddings = encoder(graph.x.to(device), graph.edge_index.to(device))

    log_probs, values_list = [], []
    placed_positions = torch.zeros(num_hard, 2)

    for macro_idx in order:
        m_emb  = embeddings[macro_idx].unsqueeze(0)
        canvas = env.canvas_tensor().to(device)

        logits, value = policy(m_emb, canvas)

        inv_mask = torch.tensor(
            env.get_invalid_mask(macro_idx), dtype=torch.bool, device=device
        )
        logits[0][inv_mask] = -1e9

        dist   = Categorical(logits=logits[0])
        action = dist.probs.argmax().item() if greedy else dist.sample().item()

        log_probs.append(dist.log_prob(torch.tensor(action, device=device)))
        values_list.append(value.squeeze())

        pos = env.step(macro_idx, action)
        placed_positions[macro_idx] = torch.tensor([pos.x, pos.y])

    reward = -evaluator_fn(bm, plc, placed_positions)
    return log_probs, values_list, reward, placed_positions

CKPT_PATH  = Path(__file__).parent / 'weights.pt'
ICCAD_ROOT = Path('external/MacroPlacement/Testcases/ICCAD04')
GRID       = 16
N_ROT      = 2
ACTION_DIM = GRID * GRID * N_ROT

class GNNSAPlacer:
    def __init__(self):
        self.device  = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.encoder = GNNEncoder().to(self.device)
        self.policy  = PlacementPolicy().to(self.device)
        if CKPT_PATH.exists():
            ckpt = torch.load(CKPT_PATH, map_location=self.device, weights_only=False)
            self.encoder.load_state_dict(ckpt['encoder'])
            self.policy.load_state_dict(ckpt['policy'])
        self.encoder.eval()
        self.policy.eval()

    def place(self, benchmark: Benchmark) -> torch.Tensor:
        # load plc for this benchmark
        from macro_place.loader import load_benchmark_from_dir
        bdir = ICCAD_ROOT / benchmark.name
        if bdir.exists():
            _, plc = load_benchmark_from_dir(str(bdir))
        else:
            return benchmark.macro_positions  # fallback

        graph = benchmark_to_graph(benchmark, plc).to(self.device)

        # GNN placement
        with torch.no_grad():
            _, _, _, gnn_pos = run_episode(
                benchmark, plc, graph,
                self.encoder, self.policy, greedy=True
            )

        # Will's SA refinement on top
        env     = PlacementEnv(benchmark, plc, evaluator_fn)
        refined = env.will_sa_refine_2phase(gnn_pos, graph, verbose=False)

        # ---- GUARANTEED final legalization ----
        n_hard   = benchmark.num_hard_macros
        W, H     = benchmark.canvas_width, benchmark.canvas_height
        sizes_np = benchmark.macro_sizes[:n_hard].numpy().astype(np.float64)
        half_w   = sizes_np[:, 0] / 2
        half_h   = sizes_np[:, 1] / 2
        if hasattr(benchmark, 'get_movable_mask'):
            movable = benchmark.get_movable_mask()[:n_hard].numpy()
        else:
            movable = ~benchmark.macro_fixed[:n_hard].numpy()

        pos = refined.numpy().astype(np.float64)
        pos = env._legalize(pos, movable, sizes_np, half_w, half_h, W, H, n_hard)
        refined = torch.tensor(pos, dtype=torch.float32)
        # ---------------------------------------

        # Return full position tensor (hard + soft)
        full_pos = benchmark.macro_positions.clone()
        full_pos[:n_hard] = refined
        return full_pos

def get_placer():
    return GNNSAPlacer()