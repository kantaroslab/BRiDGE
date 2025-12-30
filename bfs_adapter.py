# -*- coding: utf-8 -*-
"""
Created on Sat Jul 19 12:16:12 2025

@author: samar
"""

# bfs_adapter.py
# Utilities to (1) compute permanent & bridge connectivity from your Global PRM,
# (2) build the JSON snapshot your BFS expects, and (3) condense BFS output to triplets.

from collections import deque
from typing import Dict, List, Set, Tuple, Optional
import networkx as nx
import bridging_planner

try:
    from shapely.geometry import Point
except Exception:
    Point = None  # Only used for precise plane-contains if available

# --------- small helpers ------------------------------------------------------

def _canon_pair(a: str, b: str) -> Tuple[str, str]:
    return (a, b) if a <= b else (b, a)

def _pair_key(a: str, b: str) -> str:
    a, b = _canon_pair(a, b)
    return f"{a}-{b}"

def _plane_only_view(global_prm):
    """Induce a subgraph containing only nodes whose metadata marks them as Plane."""
    plane_nodes = [i for i, m in enumerate(global_prm.metadata) if m.get("source_type") == "Plane"]
    return global_prm.nx_graph.subgraph(plane_nodes).copy()

def _indices_by_plane(global_prm) -> Dict[str, List[int]]:
    """Map plane name -> list of node indices (plane-only nodes)."""
    idxs: Dict[str, List[int]] = {}
    for i, m in enumerate(global_prm.metadata):
        if m.get("source_type") == "Plane":
            pname = m.get("source")
            if pname is None:
                continue
            idxs.setdefault(pname, []).append(i)
    return idxs

def _shortest_path_between_sets(G: nx.Graph, srcs: List[int], tgts: Set[int]) -> Optional[List[int]]:
    """Unweighted multi-source BFS to any target; returns node-index path or None."""
    if not srcs or not tgts:
        return None
    q = deque()
    parent = {}
    seen = set()
    for s in srcs:
        q.append(s)
        seen.add(s)
        parent[s] = None
    while q:
        u = q.popleft()
        if u in tgts:
            # reconstruct
            path = []
            v = u
            while v is not None:
                path.append(v)
                v = parent[v]
            return list(reversed(path))
        for v in G.neighbors(u):
            if v not in seen:
                seen.add(v)
                parent[v] = u
                q.append(v)
    return None

def find_plane_for_point(xyz, planes, z_tol: float = 1e-6) -> Optional[str]:
    """
    Fast plane lookup: filter by z, then 2D contains if available.
    Returns plane.name or None if not found.
    """
    x, y, z = xyz
    # First pass: filter by z level
    candidates = []
    for pl in planes:
        # assume all vertices share the same z for a plane
        pz = pl.vertices[0][2]
        if abs(z - pz) <= z_tol:
            candidates.append(pl)
    # If only one candidate, done
    if len(candidates) == 1:
        return candidates[0].name
    # If multiple, prefer polygon contains if we have shapely
    if Point is not None and candidates:
        pt = Point(x, y)
        for pl in candidates:
            if hasattr(pl, "shape") and pl.shape is not None:
                try:
                    if pl.shape.contains(pt) or pl.shape.touches(pt):
                        return pl.name
                except Exception:
                    pass
    # Fallback: nearest by |z| (if nothing matched)
    if not candidates and len(planes) > 0:
        nearest = min(planes, key=lambda pl: abs(pl.vertices[0][2] - z))
        return nearest.name
    return candidates[0].name if candidates else None

def _support_plane_for_block(block, planes, z_tol: float = 1e-6) -> Optional[str]:
    """
    Determine the supporting plane for a block using its bottom centroid.
    """
    if hasattr(block, "centroid") and block.centroid is not None:
        return find_plane_for_point(block.centroid, planes, z_tol=z_tol)
    return None


def _to_plane_name(obj, planes) -> str:
    """Accepts plane name, Plane object, or plane index; returns plane name."""
    if isinstance(obj, str):
        return obj
    if hasattr(obj, "name"):
        return obj.name
    if isinstance(obj, int):
        # tolerate int indices
        if 0 <= obj < len(planes):
            return planes[obj].name
        return str(obj)
    return str(obj)

def _to_block_name(obj, blocks) -> str:
    """Accepts block name, Block object, or numeric index; returns block name."""
    if isinstance(obj, str):
        return obj
    if hasattr(obj, "name") and obj.name is not None:
        return obj.name
    if isinstance(obj, int):
        # map index -> blocks[index].name
        if 0 <= obj < len(blocks):
            name = getattr(blocks[obj], "name", None)
            if name is not None:
                return name
    # fallback: stringify
    return str(obj)

def _canonicalize_candidates(raw_cands, planes, blocks) -> dict:
    """
    Convert arbitrary gap keys (('Ground','Table1'), (0,1), 'Table1-Ground', etc.)
    into BFS-required form: { "Ground-Table1": [[block_name, place_plane_name], ...], ... }.
    Also maps numeric block indices to their .name.
    """
    norm: dict[str, list[list[str]]] = {}
    for gap_key, pairs in raw_cands.items():
        # canonicalize gap key to "A-B"
        if isinstance(gap_key, str):
            parts = [x.strip() for x in gap_key.split("-")] if "-" in gap_key else [gap_key]
            if len(parts) == 2:
                a, b = parts
            else:
                raise ValueError(f"Bad gap key '{gap_key}': expected 'A-B'.")
        elif isinstance(gap_key, (tuple, list)) and len(gap_key) == 2:
            a, b = gap_key
        else:
            raise ValueError(f"Unsupported gap key type: {type(gap_key)} -> {gap_key}")

        A = _to_plane_name(a, planes)
        B = _to_plane_name(b, planes)
        if B < A:
            A, B = B, A
        canon_gap = f"{A}-{B}"

        # normalize candidate value pairs to [block_name:str, place_plane_name:str]
        out_pairs: list[list[str]] = []
        for p in pairs:
            if not isinstance(p, (tuple, list)) or len(p) != 2:
                raise ValueError(f"Bad candidate pair for {canon_gap}: {p}")
            blk, place = p
            blk_name   = _to_block_name(blk, blocks)          # <-- maps int -> blocks[int].name
            place_name = _to_plane_name(place, planes)
            out_pairs.append([blk_name, place_name])

        norm.setdefault(canon_gap, []).extend(out_pairs)
    return norm


# --------- 1) PERMANENT CONNECTIVITY (compute once) ---------------------------

def compute_permanent_connectivity(global_prm, planes) -> Set[Tuple[str, str]]:
    """
    Structure-only connectivity between planes (no blocks, no third-plane redundancy).
    A pair (A,B) is permanent if there exists a path in the plane-only subgraph
    between any node on A and any node on B. From any found path, we add all
    consecutive pairs (A->C->D adds (A,C) and (C,D)), enabling the skip optimization.
    Returns: set of canonicalized pairs, e.g. {("Ground","Table1"), ...}
    """
    Gp   = _plane_only_view(global_prm)
    idxs = _indices_by_plane(global_prm)
    plane_names = [p.name for p in planes if idxs.get(p.name)]
    permanent: Set[Tuple[str, str]] = set()
    resolved:  Set[Tuple[str, str]] = set()

    # Cache plane label per node
    node_plane = {}
    for n in Gp.nodes:
        node_plane[n] = global_prm.metadata[n].get("source")

    for i in range(len(plane_names)):
        for j in range(i + 1, len(plane_names)):
            A, B = plane_names[i], plane_names[j]
            cab = _canon_pair(A, B)
            if cab in resolved:
                continue

            path_nodes = _shortest_path_between_sets(Gp, idxs[A], set(idxs[B]))
            if not path_nodes:
                continue

            # compress consecutive plane labels along the path
            seq: List[str] = []
            prev = None
            for n in path_nodes:
                pn = node_plane.get(n)
                if pn != prev:
                    seq.append(pn)
                    prev = pn

            # add all consecutive pairs from the sequence (skip trick)
            for k in range(len(seq) - 1):
                u, v = seq[k], seq[k + 1]
                cuv = _canon_pair(u, v)
                permanent.add(cuv)
                resolved.add(cuv)

    return permanent

# --------- 2) BRIDGE CONNECTIVITY (compute each call, incl. initial) ----------

def compute_bridge_connectivity(global_prm,
                                planes,
                                blocks,
                                permanent_edges: Set[Tuple[str, str]],
                                z_tol: float = 1e-6):
    """
    Detect plane pairs that are connected *via exactly one block hop* in current PRM.
    Returns:
      bridge_edges: set[(Pi,Pj)] for planes bridged now (excluding permanent)
      init_bridges: list[{ "block": Bk, "gap": "Pi-Pj", "place_plane": Pk }]
    """
    # Pre-map plane nodes and their plane name
    plane_name_of = {}
    for i, m in enumerate(global_prm.metadata):
        if m.get("source_type") == "Plane":
            pname = m.get("source")
            if pname is not None:
                plane_name_of[i] = pname

    # Build a quick reverse map from block identity to the block object
    # We'll try several attributes for robustness: uid, name
    blocks_by_uid = {}
    blocks_by_name = {}
    for b in blocks:
        if hasattr(b, "uid"):
            blocks_by_uid[b.uid] = b
        if hasattr(b, "name"):
            blocks_by_name[b.name] = b

    bridge_edges: Set[Tuple[str, str]] = set()
    init_bridges: List[dict] = []

    # Scan all nodes that are block nodes; group neighbors by plane
    for node_idx, m in enumerate(global_prm.metadata):
        if m.get("source_type") != "Block":
            continue

        # Identify the corresponding block object if possible
        blk_obj = None
        blk_name = None
        if "name" in m:
            blk_name = m["name"]
            blk_obj = blocks_by_name.get(blk_name)
        if blk_obj is None and "uid" in m:
            blk_obj = blocks_by_uid.get(m["uid"])
        if blk_name is None and blk_obj is not None and hasattr(blk_obj, "name"):
            blk_name = blk_obj.name
        if blk_name is None:
            blk_name = f"block_{node_idx}"

        # Collect distinct neighbor planes this block connects to
        neighbor_planes = set()
        for v in global_prm.nx_graph.neighbors(node_idx):
            pname = plane_name_of.get(v, None)
            if pname:
                neighbor_planes.add(pname)

        neighbor_planes = list(neighbor_planes)
        if len(neighbor_planes) < 2:
            continue

        # supporting plane (for BFS 'place_plane' field)
        place_plane = _support_plane_for_block(blk_obj, planes, z_tol=z_tol) if blk_obj else None

        # Add all unordered pairs among neighbor planes, excluding permanent ones
        for i in range(len(neighbor_planes)):
            for j in range(i + 1, len(neighbor_planes)):
                A, B = neighbor_planes[i], neighbor_planes[j]
                cuv = _canon_pair(A, B)
                if cuv in permanent_edges:
                    continue  # structurally connected already
                bridge_edges.add(cuv)
                init_bridges.append({
                    "block": blk_name,
                    "gap": _pair_key(A, B),
                    "place_plane": place_plane if place_plane is not None else A
                })

    return bridge_edges, init_bridges

# --------- 3) Snapshot builder for your BFS schema ----------------------------

def build_bfs_snapshot(planes,
                       global_prm,
                       blocks,
                       robot_xyz,
                       goal_xyz,
                       candidates: Dict[str, List[List[str]]],
                       permanent_edges: Set[Tuple[str, str]],
                       z_tol: float = 1e-6) -> dict:
    """
    Build the flat dict your BFS uses:
      {
        "planes": [...],
        "connected": [[Pi,Pj], ...],                 # permanent ∪ current bridge edges
        "start": "Pi",
        "target": "Pj",
        "blocks": ["B1","B2",...],
        "init_loc": {"B1":"Pi", ...},
        "candidates": { "Pi-Pj": [["B1","Pk"], ...], ... },
        "init_bridges": [{"block":"B2","gap":"Pi-Pj","place_plane":"Pk"}, ...]
      }
    """
    # dynamic bridges (incl. initial state)
    bridge_edges, init_bridges = compute_bridge_connectivity(
        global_prm, planes, blocks, permanent_edges, z_tol=z_tol
    )
    connected = sorted({tuple(e) for e in permanent_edges} | {tuple(e) for e in bridge_edges})

    plane_names = [p.name for p in planes]
    start_plane = find_plane_for_point(robot_xyz, planes, z_tol=z_tol)
    target_plane = find_plane_for_point(goal_xyz,  planes, z_tol=z_tol)

    # block ids and initial locations
    block_ids = []
    init_loc = {}
    for b in blocks:
        bid = getattr(b, "name", None)
        if bid is None:
            raise ValueError("Each Block must have a 'name' (e.g., 'B1','B2',...).")
        block_ids.append(bid)
        init_loc[bid] = _support_plane_for_block(b, planes, z_tol=z_tol)

    env = {
        "planes": plane_names,
        "connected": [list(e) for e in connected],
        "start": start_plane,
        "target": target_plane,
        "blocks": block_ids,
        "init_loc": init_loc,
        "candidates": _canonicalize_candidates(candidates, planes, blocks),      # superset; keep under-block exclusion in your drop sampler
        "init_bridges": init_bridges   # may be empty
    }
    return env

# --------- 4) BFS call + condense to triplets --------------------------------

def bfs_solve_to_triplets(snapshot: dict):
    """
    Calls bridging_planner.solve(snapshot) and condenses to triplets:
      (block_id, place_plane_id, gap_id_or_None)

    Supports both return shapes:
      - {"plan": [...]}                             # single plan
      - {"solutions": [{"plan_id":..., "steps":[...]} , ...]}  # multiple plans
    """

    result = bridging_planner.solve(snapshot)

    # Normalize to a list of plan steps
    plan_steps = None
    if isinstance(result, dict):
        if "plan" in result and isinstance(result["plan"], list):
            plan_steps = result["plan"]
        elif "solutions" in result and isinstance(result["solutions"], list) and result["solutions"]:
            # pick the first solution by default
            first = result["solutions"][0]
            plan_steps = first.get("steps") or first.get("plan")  # tolerate either key
    elif isinstance(result, list):
        plan_steps = result

    if not plan_steps:
        return []

    triplets = []
    for step in plan_steps:
        # Fields typically look like: {'action':'Place','block':'B1','plane':'Pk','gap':'A-B'}
        action = step.get("action") or step.get("type") or step.get("op")
        if action not in ("Place", "TempPlace"):
            continue
        blk = step.get("block")
        pln = step.get("plane") or step.get("place_plane")
        gap = step.get("gap")  # keep None if not explicitly bridging
        triplets.append((blk, pln, gap))

    return triplets

def triplets_to_bias_sequence(triplets, planes, blocks):
    """
    Convert name-based triplets (blk_name, place_plane_name, gap_str_or_None)
    into your bias sequence format:
        [ ((gap_i, gap_j) or None, block_idx, place_plane_idx), ... ]

    - planes: either a list of Plane objects (with .name) OR a list of plane names (str)
    - blocks: either a list of Block objects (with .name) OR a list of block names (str)
    """
    # Build name -> index maps
    if planes and hasattr(planes[0], "name"):
        plane_names = [p.name for p in planes]
    else:
        plane_names = list(planes)
    plane_idx = {name: i for i, name in enumerate(plane_names)}

    if blocks and hasattr(blocks[0], "name"):
        block_names = [b.name for b in blocks]
    else:
        block_names = list(blocks)
    block_idx = {name: i for i, name in enumerate(block_names)}

    seq = []
    for blk_name, place_plane_name, gap in triplets:
        # block index
        if blk_name not in block_idx:
            raise KeyError(f"Unknown block name in triplet: {blk_name}")
        b_i = block_idx[blk_name]

        # place plane index
        if place_plane_name not in plane_idx:
            raise KeyError(f"Unknown plane name in triplet: {place_plane_name}")
        p_i = plane_idx[place_plane_name]

        # gap indices (or None if random/temp placement)
        if gap is None:
            gap_pair = None
        else:
            a, b = [t.strip() for t in gap.split("-")]
            if a not in plane_idx or b not in plane_idx:
                raise KeyError(f"Unknown plane in gap '{gap}'")
            i, j = plane_idx[a], plane_idx[b]
            gap_pair = (i, j) if i <= j else (j, i)

        seq.append((gap_pair, b_i, p_i))

    return seq

