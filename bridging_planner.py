#!/usr/bin/env python3
import sys, json
from collections import deque, defaultdict
from copy import deepcopy

def normalize_gap(p1, p2):
    a, b = sorted([p1, p2])
    return f"{a}-{b}"

def _canon_gap_string(gap_str):
    p1, p2 = [x.strip() for x in gap_str.split("-")]
    return normalize_gap(p1, p2)


def build_env(raw_env):
    planes = set(raw_env["planes"])
    start = raw_env["start"]
    target = raw_env["target"]
    blocks = list(raw_env["blocks"])
    init_loc = dict(raw_env["init_loc"])

    base_edges = defaultdict(set)
    for u, v in raw_env.get("connected", []):
        base_edges[u].add(v); base_edges[v].add(u)

    raw_cands = raw_env["candidates"]
    candidates = defaultdict(list)
    gap_label = {}  # canonical gap id -> first-seen original label
    for gap_key, pairs in raw_cands.items():
        p1, p2 = [x.strip() for x in gap_key.split("-")]
        canon = normalize_gap(p1, p2)
        if canon not in gap_label:
            gap_label[canon] = gap_key  # preserve original string format
        for (blk, place_plane) in pairs:
            candidates[canon].append((blk, place_plane))

    env = {
        "planes": planes, "base_edges": base_edges,
        "start": start, "target": target, "blocks": blocks,
        "init_loc": init_loc, "candidates": dict(candidates),
        "gap_label": gap_label
    }
    env["init_bridges"] = []
    for ent in raw_env.get("init_bridges", []):
        # ent expects: {"block": "B2", "gap": "A-B", "place_plane": "A"}
        b = ent["block"]; gap_raw = ent["gap"]; plc = ent["place_plane"]
        canon = _canon_gap_string(gap_raw)
        env["init_bridges"].append({"block": b, "canon_gap": canon, "place_plane": plc, "gap_raw": gap_raw})
    return env

def initial_state(env):
    # default: every block lying on its plane (not temp-placed)
    loc = {b: ("PLANE", p, False) for b, p in env["init_loc"].items()}
    # override for any blocks that start as active bridges
    for info in env.get("init_bridges", []):
        b = info["block"]
        canon = info["canon_gap"]
        plc = info["place_plane"]
        # Note: we store plane of pickup for a bridge in the 3rd slot
        loc[b] = ("BRIDGE", canon, plc)
    return {"robot": env["start"], "held": None, "loc": loc}

def bridged_edges(state):
    edges = set()
    for info in state["loc"].values():
        if info[0] == "BRIDGE":
            a, b = info[1].split("-")
            edges.add((a, b)); edges.add((b, a))
    return edges

def neighbors(robot, env, state):
    N = set(env["base_edges"].get(robot, set()))
    for u, v in bridged_edges(state):
        if u == robot: N.add(v)
    return N

def canonicalize(env, state):
    key_loc = {}
    for b in env["blocks"]:
        info = state["loc"][b]
        if info[0] == "PLANE": key_loc[b] = ("PLANE", info[1])
        elif info[0] == "BRIDGE": key_loc[b] = ("BRIDGE", info[1])
        else: key_loc[b] = ("HELD",)
    return (state["robot"], state["held"], tuple(sorted(key_loc.items())))

def generate_actions(env, state):
    A = []
    for p in neighbors(state["robot"], env, state):
        A.append(("GoTo", {"plane": p}))
    if state["held"] is None:
        for b in env["blocks"]:
            info = state["loc"][b]
            if info[0] == "PLANE" and info[1] == state["robot"]:
                A.append(("TempPick" if info[2] else "Pick", {"block": b}))
            elif info[0] == "BRIDGE" and info[2] == state["robot"]:
                A.append(("Pick", {"block": b}))
    if state["held"] is not None:
        B = state["held"]
        already_bridged = {i[1] for i in state["loc"].values() if i[0] == "BRIDGE"}
        for g, cand_list in env["candidates"].items():
            if g in already_bridged: continue
            for (blk, place_plane) in cand_list:
                if blk == B and place_plane == state["robot"]:
                    A.append(("Place", {"block": B, "plane": state["robot"], "gap": env["gap_label"][g]}))
                    break
        A.append(("TempPlace", {"block": B, "plane": state["robot"]}))
    return A

def apply_action(env, state, action):
    t, args = action
    s = deepcopy(state)
    if t == "GoTo":
        p = args["plane"]
        assert p in neighbors(s["robot"], env, s)
        s["robot"] = p; return s
    if t in ("Pick", "TempPick"):
        b = args["block"]; assert s["held"] is None
        info = s["loc"][b]
        if info[0] == "PLANE": assert info[1] == s["robot"]
        elif info[0] == "BRIDGE": assert info[2] == s["robot"]
        else: raise AssertionError
        s["loc"][b] = ("HELD",); s["held"] = b; return s
    if t == "TempPlace":
        b = args["block"]; assert s["held"] == b
        s["loc"][b] = ("PLANE", s["robot"], True); s["held"] = None; return s
    if t == "Place":
        b = args["block"]; plane = args["plane"]; gap_str = args["gap"]
        # map back to canonical
        p1, p2 = [x.strip() for x in gap_str.split("-")]
        canon = normalize_gap(p1, p2)
        assert s["held"] == b and plane == s["robot"]
        ok = any((blk == b and place_plane == plane) for blk, place_plane in env["candidates"].get(canon, []))
        assert ok
        s["loc"][b] = ("BRIDGE", canon, plane); s["held"] = None; return s
    raise ValueError

def format_action(a):
    t, args = a; out = {"action": t if t != "TempPick" else "TempPick"}
    if t in ("Pick", "TempPick"): out["block"] = args["block"]
    elif t == "GoTo": out["plane"] = args["plane"]
    elif t == "TempPlace": out["block"] = args["block"]; out["plane"] = args["plane"]
    elif t == "Place": out["block"] = args["block"]; out["plane"] = args["plane"]; out["gap"] = args["gap"]
    return out


def find_one_plan(env):
    start = initial_state(env)
    if env["start"] == env["target"]: return []
    q = deque([start]); visited = {canonicalize(env, start)}; parent = {canonicalize(env, start):(None,None)}
    
    # (optional) counters
    expanded = 0
    enqueued = 0

    while q:
        s = q.popleft()
        expanded += 1  # optional
        if s["robot"] == env["target"]:
            # >>> single print you asked for (unique states discovered so far)
            print("[BFS] unique states discovered: ", expanded, 
                 "; enqueued: " , enqueued)  # expanded/enqueued are optional

            
            steps = []; key = canonicalize(env, s)
            while True:
                prev = parent.get(key)
                if prev is None: break
                prev_key, action = prev
                if action is None: break
                steps.append(format_action(action)); key = prev_key
            return list(reversed(steps))
        for a in generate_actions(env, s):
            try: s2 = apply_action(env, s, a)
            except AssertionError: continue
            key = canonicalize(env, s2)
            if key not in visited:
                visited.add(key); parent[key]=(canonicalize(env, s), a); q.append(s2)
                enqueued += 1  # optional
    return None

def enumerate_plans_bfs(env, K=10, include_longer=False, depth_limit=None, multiparents=False):
    """
    Return up to K plans in non-decreasing length using BFS layers.
    - include_longer=False  -> only shortest-length plans
      include_longer=True   -> keep going to next depths until K plans
    - multiparents=False    -> one parent per state (one path per terminal state)
      multiparents=True     -> store all parents at same best depth (can explode)
    """
    s0 = initial_state(env)
    k0 = canonicalize(env, s0)

    # BFS frontier over (state, depth)
    q = deque([(s0, 0)])
    best_depth = {k0: 0}

    if multiparents:
        parents = defaultdict(list)   # key -> list of (prev_key, action)
        parents[k0] = []              # start has no parents
    else:
        parents = {k0: (None, None)}  # key -> (prev_key, action)

    goal_keys = []
    first_goal_depth = None

    while q and len(goal_keys) < K:
        s, d = q.popleft()
        k = canonicalize(env, s)

        # Goal test
        if s["robot"] == env["target"]:
            if first_goal_depth is None:
                first_goal_depth = d
            # Only accept goals at first_goal_depth unless include_longer=True
            if include_longer or d == first_goal_depth:
                goal_keys.append(k)
                if not include_longer and len(goal_keys) >= K:
                    break
            # If we already passed the shortest layer and we're not including longer plans, stop
            elif not include_longer and d > first_goal_depth:
                break
            # We still continue expanding queue items at same depth to collect all shortest goals
            continue

        if depth_limit is not None and d >= depth_limit:
            continue

        # Expand
        for a in generate_actions(env, s):
            try:
                s2 = apply_action(env, s, a)
            except AssertionError:
                continue
            k2 = canonicalize(env, s2)
            nd = d + 1
            bd = best_depth.get(k2)

            # Standard BFS rule + optional multi-parent at equal best depth
            if bd is None or nd < bd or (multiparents and nd == bd):
                best_depth[k2] = nd
                if multiparents:
                    parents[k2].append((k, a))
                else:
                    parents[k2] = (k, a)
                q.append((s2, nd))

    # ---- Reconstruction helpers ----
    def reconstruct_one_path(k_goal):
        # Single-parent backtrace
        path = []
        k = k_goal
        while True:
            prev = parents.get(k)
            if not prev: break
            pk, a = prev
            if a is None: break
            path.append(format_action(a))
            k = pk
        return list(reversed(path))

    def reconstruct_all_paths(k_goal, Kleft):
        # Multi-parent backtracking (all shortest, capped by Kleft)
        results = []

        def dfs(k, acc_rev):
            if not parents[k]:  # reached start
                results.append(list(reversed(acc_rev)))
                return len(results) >= Kleft
            for pk, a in parents[k]:
                acc_rev.append(format_action(a))
                stop = dfs(pk, acc_rev)
                acc_rev.pop()
                if stop:
                    return True
            return False

        dfs(k_goal, [])
        return results[:Kleft]

    # ---- Collect plans ----
    plans = []
    if multiparents:
        kleft = K
        for k in goal_keys:
            if kleft <= 0: break
            chunk = reconstruct_all_paths(k, kleft)
            plans.extend(chunk)
            kleft -= len(chunk)
    else:
        for i, k in enumerate(goal_keys):
            plans.append(reconstruct_one_path(k))
            if len(plans) >= K: break

    return [{"plan_id": i+1, "steps": p} for i, p in enumerate(plans)]


def enumerate_plans(env, K_max=5, depth_cap=None):
    first = find_one_plan(env)
    if first is None: return []
    shortest_len = len(first); 
    if depth_cap is None: depth_cap = shortest_len + 6
    results = [first]
    init = initial_state(env)
    stack = [(init, [], {canonicalize(env, init)})]
    while stack and len(results) < K_max:
        s, prefix, seen = stack.pop()
        if s["robot"] == env["target"]:
            if prefix and prefix not in results: results.append(prefix)
            continue
        if len(prefix) >= depth_cap: continue
        for a in generate_actions(env, s):
            try: s2 = apply_action(env, s, a)
            except AssertionError: continue
            key2 = canonicalize(env, s2)
            if key2 in seen: continue
            stack.append((s2, prefix + [format_action(a)], seen | {key2}))
    return results

def solve(env_dict, enumerate_all=False, K=5, include_longer=False, multiparents=False):
    env = build_env(env_dict)
    if enumerate_all:
        # plans = enumerate_plans(env, K_max=K)
        # return {"solutions": [{"plan_id": i+1, "steps": plan} for i, plan in enumerate(plans)]}
        sols = enumerate_plans_bfs(env, K=K, include_longer=include_longer, multiparents=multiparents)
        return {"solutions": sols}
    else:
        plan = find_one_plan(env)
        if plan is None: return {"solutions": []}
        return {"solutions": [{"plan_id": 1, "steps": plan}]}

def _cli():
    try:
        raw = sys.stdin.read().strip()
        if not raw: print('{"solutions": []}'); return
        env = json.loads(raw)
        res = solve(env, enumerate_all=False, K=1)
        print(json.dumps(res, indent=2))
    except Exception as e:
        print(json.dumps({"error": str(e)}))

if __name__ == "__main__":

    # # Your test case
    # env_dict = {
    #   "environment": {
    #     "planes": ["G", "B1", "B2", "O", "Y", "R", "P"],
    #     "connected": [["B1", "O"]],
    #     "start": "G",
    #     "target": "P",
    #     "blocks": ["1", "2", "3", "4", "5"],
    #     "init_loc": {
    #       "1": "G",
    #       "2": "B1",
    #       "3": "R",
    #       "4": "R",
    #       "5": "Y"
    #     },
    #     "candidates": {
    #       "G-B1": [["1", "G"], ["2", "G"]],
    #       "G-B2": [["1", "G"], ["2", "G"]],
    #       "B2-R": [["1", "B2"], ["2", "B2"]],
    #       "O-Y":  [["3", "B1"], ["4", "B1"]],
    #       "B2-P": [["5", "G"]],
    #       "R-P":  [["5", "G"]]
    #     }
    #   }
    # }
    
    env_dict = {
      "environment": {
        "planes": ["G", "Table1", "Table2", "O", "Y", "R"],
        "connected": [["Table1", "O"]],
        "start": "G",
        "target": "Y",
        "blocks": ["1", "2", "3"],
        "init_loc": {
          "1": "G",
          "2": "Table1",
          "3": "R",
        },
        "candidates": {
          "G-Table1": [["1", "G"], ["2", "G"]],
          "G-Table2": [["1", "G"], ["2", "G"]],
          "Table2-R": [["2", "Table2"],["1", "Table2"]],
          "O-Y":  [["3", "Table1"]]
        }
      }
    }
    
    # env_dict = {
    #   "environment":  {
    #       'planes': ['Ground', 'Table1', 'Table2', 'O', 'Y', 'R'], 
    #       'connected': [['O', 'Table1']], 
    #       'start': 'Ground', 
    #       'target': 'Y', 
    #       'blocks': ['B1', 'B2', 'B3'], 
    #       'init_loc': {'B1': 'Ground', 'B2': 'Table1', 'B3': 'R'}, 
    #       'candidates': {
    #           'Ground-Table1': [['B1', 'Ground'], ['B2', 'Ground']], 
    #           'Ground-Table2': [['B1', 'Ground'], ['B2', 'Ground']], 
    #           'O-Y': [['B3', 'Table1']], 
    #           'R-Table2': [['B2', 'Table2']]}, 
    #       'init_bridges': []}
    # }
    
    # env_dict = {
    #   "environment": {
    #     "planes": ["G", "B1", "B2", "O", "Y", "R"],
    #     "connected": [["B1", "O"]],
    #     "start": "G",
    #     "target": "R",
    #     "blocks": ["1", "2", "3"],
    #     "init_loc": {
    #       "1": "G",
    #       "2": "B1",
    #       "3": "R",
    #     },
    #     "candidates": {
    #       "G-B1": [["1", "G"], ["2", "G"]],
    #       "G-B2": [["1", "G"], ["2", "G"]],
    #       "B2-R": [["2", "B2"]],
    #       "O-Y":  [["3", "B1"]]
    #     }
    #   }
    # }
    
    # env_dict = {
    #   "environment": {
    #       'planes': ['Ground', 'Table1', 'Table2'], 
    #       'connected': [], 
    #       'start': 'Ground', 
    #       'target': 'Table2', 
    #       'blocks': ['B1', 'B2'], 
    #       'init_loc': {'B1': 'Ground', 'B2': 'Ground'}, 
          # 'candidates': {
          #     'Ground-Table1': [['B1', 'Ground'], ['B2', 'Ground']],
          #     'Table1-Table2': [['B1', 'Table1'], ['B2', 'Table1']]
    #           }, 
    #       'init_bridges': []}
    # }
    
    
    # env_dict = {
    #   "environment": {
    #     "planes": ["G", "B1", "B2"],
    #     "connected": [],
    #     "start": "G",
    #     "target": "B1",
    #     "blocks": ["1", "2"],
    #     "init_loc": {
    #       "1": "G",
    #       "2": "G",
    #     },
    #     "candidates": {
    #       "G-B1": [["1", "G"], ["2", "G"]],
    #       "G-B2": [["1", "G"], ["2", "G"]],
    #     }
    #   }
    # }



    # Run: one shortest plan
    one = solve(env_dict["environment"], enumerate_all=False)
    print("One shortest plan:")
    print(json.dumps(one, indent=2))


    # Run: enumerate a few alternatives (increase K if you want more)
    # many = solve(env_dict["environment"], enumerate_all=True, K=50)
    # print("\nUp to 5 plans:")
    # print(json.dumps(many, indent=2))
