# -*- coding: utf-8 -*-
"""
Created on Sat Jul 26 14:03:09 2025

@author: AsuS
"""

"""
LLM Adapter for High-Level Bridge Planning
==========================================

This module defines a thin wrapper that:
  1. Takes the same symbolic snapshot used by the BFS planner
     (see bfs_adapter.build_bfs_snapshot).
  2. Builds a text prompt for an LLM.
  3. Expects the LLM to return a JSON object containing a single
     high-level plan in the following format:
        {
          "solutions": [
            {
              "plan_id": <int>,
              "steps": [
                { "action": "GoTo",      "plane": "<PlaneID>"              },
                { "action": "Pick",      "block": "<BlockID>"              },
                { "action": "Place",     "block": "<BlockID>",
                                         "plane": "<CandidatePlane>",
                                         "gap":   "<PlaneA>-<PlaneB>"     },
                { "action": "TempPlace", "block": "<BlockID>",
                                         "plane": "<PlaneID>"             },
                { "action": "TempPick",  "block": "<BlockID>"              }
                // …etc.
              ]
            }
          ]
        }

The core prompt we use (for documentation purposes) is:

Task:
Generate all high-level action sequences that let a robot move from a Start plane to a Target plane by using blocks to bridge gaps. The blocks can be reused.

1. Environment
    • Planes: surfaces (e.g. ground, table tops). Robot can travel anywhere within each plane.
    • Blocks: objects used as stepping-stones between planes.
    • Connected planes: robot can jump directly without the help of a block.
    • Gaps: Two planes that are not directly connected.

2. Bridging Gaps
For each gap (P1-P2) between two planes P1,P2, you’re given a list of candidate (Block, Plane) pairs—meaning “if you place that Block on that Plane, you bridge (P1-P2).”

3. Actions
    • GoTo(P): move to plane P (requires P be connected or bridged to current plane).
    • Pick(B): pick up block B (must be on current plane).
    • Place(B, P, (P1-P2)): place block B on plane P to bridge gap (P1-P2).
    • TempPlace(B, P): temporarily drop B on P (to go fetch another block).
    • TempPick(B): pick up B from its temporary drop.

4. Constraints
    • Ignore exact block positions on a plane.
    • Blocks can be reused; carry one at a time.
    • Multiple valid sequences may exist,
      BUT IN THIS TASK RETURN EXACTLY ONE SOLUTION.
    • Return ONLY the JSON object described below. Do not add explanations.

5. Input:
We describe the environment as JSON:

  {
    "environment": {
      "planes": ["<P1>", "<P2>", "...", "<Pn>"],
      "connected": [
        ["<Pi>","<Pj>"],
        ["<Pk>","<Pl>"]
        // …
      ],
      "start": "<StartPlane>",
      "target": "<TargetPlane>",
      "blocks": ["<B1>", "<B2>", "...", "<Bm>"],
      "init_loc": {
        "<B1>": "<PlaneName>",
        "<B2>": "<PlaneName>"
        // …
      },
      "candidates": {
        "<PlaneA>-<PlaneB>": [
          ["<BlockX>", "<CandidatePlane>"],
          ["<BlockY>", "<CandidatePlane>"]
        ],
        "<PlaneC>-<PlaneD>": [
          ["<BlockZ>", "<CandidatePlane>"]
        ]
        // …
      }
    }
  }

Here, planes, blocks, connected, init_loc, and candidates have the same
meaning as in bfs_adapter.build_bfs_snapshot.

6. Output format:
The LLM must respond with a JSON object of the form:

{
  "solutions": [
    {
      "plan_id": <integer>,
      "steps": [
        { "action": "GoTo",      "plane": "<PlaneID>"              },
        { "action": "Pick",      "block": "<BlockID>"              },
        { "action": "Place",     "block": "<BlockID>",
                                 "plane": "<CandidatePlane>",
                                 "gap":   "<PlaneA>-<PlaneB>"     },
        { "action": "TempPlace", "block": "<BlockID>",
                                 "plane": "<PlaneID>"              },
        { "action": "TempPick",  "block": "<BlockID>"              }
        // …etc., in the exact order to reach Target
      ]
    }
  ]
}

Example:
Prompt: “{
  "environment": {
    "planes": ["A","B","C","D"],
    "connected": [["A","B"]],
    "start": "A",
    "target": "D",
    "blocks": ["1","2","3"],
    "init_loc": {
      "1": "B",
      "2": "C",
      "3": "B"
    },
    "candidates": {
      "A-C": [["1","A"]],
      "A-D": [["2","A"]],
      "C-D": [["3","A"]]
    }
  }
}”

Output:
{
  "solutions": [
    {
      "plan_id": 1,
      "steps": [
        { "action": "GoTo",     "plane": "B"            },
        { "action": "Pick",     "block": "1"            },
        { "action": "GoTo",     "plane": "A"            },
        { "action": "Place",    "block": "1", "plane": "A", "gap": "A-C" },
        { "action": "GoTo",     "plane": "C"            },
        { "action": "Pick",     "block": "2"            },
        { "action": "GoTo",     "plane": "A"            },
        { "action": "Place",    "block": "2", "plane": "A", "gap": "A-D" },
        { "action": "GoTo",     "plane": "D"            }
      ]
    } 
  ]
}

We always use ONLY THE FIRST (AND EXPECTEDLY ONLY) SOLUTION in "solutions".

This module then extracts from that JSON the condensed triplet sequence:
    (block_id, place_plane_id, gap_id_or_None)
which has the same format as bfs_adapter.bfs_solve_to_triplets.

We re-use bfs_adapter.triplets_to_bias_sequence(...) to map these triplets
to the index-based bias sequence consumed by the sampling-based planner.
"""



import os
import json
from typing import Any, Dict, List, Tuple, Optional

from openai import OpenAI
from bfs_adapter import triplets_to_bias_sequence


# ============================================================
#  LLM PROMPT CONSTRUCTION
# ============================================================

# This is the *actual* prompt text that goes to the LLM,
# minus the concrete environment, which we append as JSON.
_BASE_PROMPT = """Task:
Generate all high-level action sequences that let a robot move from a Start plane to a Target plane by using blocks to bridge gaps. The blocks can be reused.

1. Environment
    • Planes: surfaces (e.g. ground, table tops). The robot can travel anywhere within each plane.
    • Blocks: objects used as stepping-stones between planes.
    • Connected planes: planes between which the robot can move directly (jump) without using a block.
    • Gaps: unordered pairs of planes (P1-P2) that are not directly connected.

2. Bridging Gaps
For each gap (P1-P2) between planes P1 and P2, you are given a list of candidate (Block, Plane) pairs. A pair ["Bi","Pk"] in the candidates for gap "P1-P2" means:
    "If you place block Bi on plane Pk, you create a bridge that lets the robot move between planes P1 and P2."

3. Actions
The robot can perform the following high-level actions:
    • GoTo(P): move to plane P (requires P be connected or already bridged to the current plane).
    • Pick(B): pick up block B (must be on the current plane).
    • Place(B, P, (P1-P2)): place block B on plane P to bridge the gap (P1-P2).
    • TempPlace(B, P): temporarily place/drop block B on plane P (not necessarily bridging a gap).
    • TempPick(B): pick block B back up from a temporary placement on the current plane.

4. Constraints
    • Ignore exact block positions on a plane; only the support plane matters.
    • The robot can carry at most one block at a time.
    • Blocks can be reused multiple times (pick them up again after placing).
    • There may be many valid solutions, but in this task you MUST RETURN EXACTLY ONE solution.
    • Your response MUST BE VALID JSON ONLY, with no extra text or explanation.

5. Input description
You will be given a JSON object of the form:
{
  "environment": {
    "planes": ["<P1>", "<P2>", "...", "<Pn>"],
    "connected": [
      ["<Pi>", "<Pj>"],
      ["<Pk>", "<Pl>"]
      // ...
    ],
    "start": "<StartPlane>",
    "target": "<TargetPlane>",
    "blocks": ["<B1>", "<B2>", "...", "<Bm>"],
    "init_loc": {
      "<B1>": "<PlaneName>",
      "<B2>": "<PlaneName>"
      // ...
    },
    "candidates": {
      "<PlaneA>-<PlaneB>": [
        ["<BlockX>", "<CandidatePlane>"],
        ["<BlockY>", "<CandidatePlane>"]
      ],
      "<PlaneC>-<PlaneD>": [
        ["<BlockZ>", "<CandidatePlane>"]
      ]
      // ...
    }
  }
}

6. Output JSON format
You MUST return a JSON object exactly of the following form:

{
  "solutions": [
    {
      "plan_id": <integer>,
      "steps": [
        { "action": "GoTo",      "plane": "<PlaneID>"              },
        { "action": "Pick",      "block": "<BlockID>"              },
        { "action": "Place",     "block": "<BlockID>",
                                 "plane": "<CandidatePlane>",
                                 "gap":   "<PlaneA>-<PlaneB>"     },
        { "action": "TempPlace", "block": "<BlockID>",
                                 "plane": "<PlaneID>"             },
        { "action": "TempPick",  "block": "<BlockID>"              }
        // …etc., in the exact order needed to move the robot from Start to Target
      ]
    }
  ]
}

Example:
Prompt: “{
  "environment": {
    "planes": ["A","B","C","D"],
    "connected": [["A","B"]],
    "start": "A",
    "target": "D",
    "blocks": ["1","2","3"],
    "init_loc": {
      "1": "B",
      "2": "C",
      "3": "B"
    },
    "candidates": {
      "A-C": [["1","A"]],
      "A-D": [["2","A"]],
      "C-D": [["3","A"]]
    }
  }
}”

Output:
{
  "solutions": [
    {
      "plan_id": 1,
      "steps": [
        { "action": "GoTo",     "plane": "B"            },
        { "action": "Pick",     "block": "1"            },
        { "action": "GoTo",     "plane": "A"            },
        { "action": "Place",    "block": "1", "plane": "A", "gap": "A-C" },
        { "action": "GoTo",     "plane": "C"            },
        { "action": "Pick",     "block": "2"            },
        { "action": "GoTo",     "plane": "A"            },
        { "action": "Place",    "block": "2", "plane": "A", "gap": "A-D" },
        { "action": "GoTo",     "plane": "D"            }
      ]
    } 
  ]
}

There must be exactly one object in the "solutions" array. We will use ONLY the first (and only) plan.

Now generate a solution for the following problem:
"""

def build_llm_prompt(snapshot: Dict[str, Any]) -> str:
    """
    Build the full text prompt given the BFS snapshot dictionary.

    `snapshot` is expected to be the dict returned by bfs_adapter.build_bfs_snapshot:
        {
          "planes": [...],
          "connected": [...],
          "start": "...",
          "target": "...",
          "blocks": [...],
          "init_loc": {...},
          "candidates": {...},
          "init_bridges": [...]
        }
    """
    env_wrapper = {"environment": snapshot}
    env_json = json.dumps(env_wrapper, indent=2)
    return _BASE_PROMPT + env_json + "\n"


# ============================================================
#  REAL LLM CALL (ChatGPT-5.1 via OpenAI API)
# ============================================================

def call_llm(prompt: str,
             model: str = "gpt-5.1",
             temperature: float = 0.0,
             max_tokens: int = 2048) -> str:
    """
    Call the OpenAI Chat Completions API with GPT-5.1.

    To use this, you MUST have:
      - `pip install openai`
      - an environment variable OPENAI_API_KEY set, OR you can
        edit the fallback string below.

    The function returns the raw string content of the assistant's reply,
    which should be a JSON object as described in _BASE_PROMPT.
    """
    api_key = os.getenv("OPENAI_API_KEY", "REPLACE_WITH_YOUR_OPENAI_KEY")

    client = OpenAI(api_key=api_key)

    resp = client.chat.completions.create(
        model=model,
        messages=[
            {
                "role": "system",
                "content": (
                    "You are a high-level symbolic planner that strictly outputs "
                    "valid JSON plans following the specified schema. "
                    "Return exactly one solution in the 'solutions' array and "
                    "do not include any explanations or extra text."
                ),
            },
            {
                "role": "user",
                "content": prompt,
            },
        ],
        temperature=temperature,
        max_tokens=max_tokens,
    )

    return resp.choices[0].message.content


# ============================================================
#  PARSING UTILITIES
# ============================================================

def _safe_json_parse(raw: str) -> Any:
    """
    Parse raw LLM output into JSON, tolerating leading/trailing junk if any.
    We *tell* the model to output pure JSON, but this makes the code more robust.
    """
    raw = raw.strip()
    try:
        return json.loads(raw)
    except json.JSONDecodeError:
        first = raw.find("{")
        last = raw.rfind("}")
        if first != -1 and last != -1 and last > first:
            snippet = raw[first:last+1]
            return json.loads(snippet)
        raise


def _extract_triplets_from_result(result: Any) -> List[Tuple[str, str, Optional[str]]]:
    """
    Convert the parsed JSON result into a list of triplets:

        (block_id, place_plane_id, gap_id_or_None)

    This mirrors bfs_adapter.bfs_solve_to_triplets:
      - we look at the first element of "solutions"
      - we take its "steps" list
      - and we only keep Place / TempPlace actions.
    """
    if not isinstance(result, dict):
        raise ValueError("LLM result must be a JSON object (dict).")

    sols = result.get("solutions")
    if not sols or not isinstance(sols, list):
        raise ValueError("LLM result JSON missing 'solutions' list.")

    first_sol = sols[0]
    steps = first_sol.get("steps") or first_sol.get("plan")
    if not steps or not isinstance(steps, list):
        raise ValueError("First solution does not contain a 'steps' list.")

    triplets: List[Tuple[str, str, Optional[str]]] = []
    for step in steps:
        # same logic as bfs_solve_to_triplets: we only care about Place / TempPlace
        action = step.get("action") or step.get("type") or step.get("op")
        if action not in ("Place", "TempPlace"):
            continue
        blk = step.get("block")
        pln = step.get("plane") or step.get("place_plane")
        gap = step.get("gap")  # may be None
        triplets.append((blk, pln, gap))

    return triplets


# ============================================================
#  PUBLIC ENTRYPOINTS
# ============================================================

def llm_solve_to_triplets(snapshot: Dict[str, Any],
                          raw_response: Optional[str] = None,
                          model: str = "gpt-5.1") -> List[Tuple[str, str, Optional[str]]]:
    """
    High-level function that:
      1) Builds an LLM prompt from the BFS snapshot.
      2) Either:
         - uses the provided raw_response string (for reproducibility), or
         - calls the LLM to get a fresh response.
      3) Parses the JSON.
      4) Returns the triplet list (blk_name, place_plane_name, gap_str_or_None),
         identical in format to bfs_adapter.bfs_solve_to_triplets.
    """
    if raw_response is None:
        prompt = build_llm_prompt(snapshot)
        raw_response = call_llm(prompt, model=model)

    result = _safe_json_parse(raw_response)
    return _extract_triplets_from_result(result)


def llm_triplets_to_bias_sequence(snapshot: Dict[str, Any],
                                  planes,
                                  blocks,
                                  raw_response: Optional[str] = None,
                                  model: str = "gpt-5.1"):
    """
    Convenience wrapper:
      - Compute LLM triplets from snapshot.
      - Convert them to the index-based bias sequence via
        bfs_adapter.triplets_to_bias_sequence.

    Returns:
      (triplets, bias_seq_idx)
    """
    triplets = llm_solve_to_triplets(snapshot, raw_response=raw_response, model=model)
    bias_seq_idx = triplets_to_bias_sequence(triplets, planes, blocks)
    return triplets, bias_seq_idx


