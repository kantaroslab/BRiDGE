#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Feb 19 05:22:47 2025

@author: samarth
"""


import math
import random
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Polygon as MplPolygon
from shapely.geometry import Polygon, Point, LineString
import networkx as nx
from scipy.spatial.distance import cdist
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
import time
import copy
from collections import deque
from bfs_adapter import compute_permanent_connectivity
from bfs_adapter import build_bfs_snapshot, bfs_solve_to_triplets, triplets_to_bias_sequence
# from llm_adapter import llm_solve_to_triplets  # or llm_triplets_to_bias_sequence
import proposal_2_visualizer_patched as viz


random.seed(42)

# -----------------------------
# Global sampling bias controls
# -----------------------------
UNIFORM_MODE = False  # <— flip this to False to restore biased planning
FREEZE_BFS_PLAN = False   # for the wrong-candidate experiment


p_eps = 1e-12
p_node_bias      = 0.0 if UNIFORM_MODE else 0.8



# p_triplet_plan   = p_eps if UNIFORM_MODE else 0.1
# p_triplet_gap    = 0.98 if UNIFORM_MODE else 0.6
# p_triplet_rand   = 0.02  if UNIFORM_MODE else 0.3


# p_triplet_plan   = p_eps if UNIFORM_MODE else 0.5
# p_triplet_gap    = 0.98 if UNIFORM_MODE else 0.3
# p_triplet_rand   = 0.02  if UNIFORM_MODE else 0.2



p_triplet_plan   = p_eps if UNIFORM_MODE else 0.8
p_triplet_gap    = 0.98 if UNIFORM_MODE else 0.15
p_triplet_rand   = 0.02  if UNIFORM_MODE else 0.05


MAX_ONPLAN_FAILS_BEFORE_REPLAN = float('inf') if UNIFORM_MODE else 3
TRIPLET_FAIL_THRESHOLD         = float('inf') if UNIFORM_MODE else 5

BFS_counter=1


USE_LLM_BIAS = False  # flip this to True when you want LLM bias

def edge_intersects_block(pos1, pos2, blocks):
    """
    Check if the 2D line (X, Y) connecting pos1 and pos2 intersects any block's bounding box.
    
    :param pos1: (x, y, z) of first node.
    :param pos2: (x, y, z) of second node.
    :param blocks: List of Block objects.
    :return: True if the line intersects any block bounding box, False otherwise.
    """
    line = LineString([(pos1[0], pos1[1]), (pos2[0], pos2[1])])
    for block in blocks:
        if line.intersects(block.bounding_box):
            return True
    return False


def _bbox_expand(bounds, margin):
    minx, miny, maxx, maxy = bounds
    return (minx - margin, miny - margin, maxx + margin, maxy + margin)

def _points_in_bbox_mask(xy, bbox):
    minx, miny, maxx, maxy = bbox
    return (xy[:,0] >= minx) & (xy[:,0] <= maxx) & (xy[:,1] >= miny) & (xy[:,1] <= maxy)

def _z_equal_mask(zs, z0, tol=1e-6):
    return np.abs(zs - z0) < tol

# ------------------------------------------
# 1. Robot Class
# ------------------------------------------

class Robot:
    def __init__(self, position, jump_height, horizontal_range, move_range, reach_range):
        """
        Initialize a robot with specific dynamics.
        
        :param position: Tuple (x, y, z) representing the initial position.
        :param jump_height: Maximum vertical distance the robot can jump.
        :param horizontal_range: Maximum horizontal distance the robot can cover in one jump.
        :param move_range: Maximum distance the robot can move in one time step.
        :param reach_range: Maximum distance the robot's arm can reach.
        """
        self.position = position
        self.jump_height = jump_height
        self.horizontal_range = horizontal_range
        self.move_range = move_range
        self.reach_range = reach_range

    def can_jump(self, start, end):
        """
        Check if the robot can jump from start to end based on its jump dynamics.
        
        :param start: Tuple (x, y, z) of the starting position.
        :param end: Tuple (x, y, z) of the target position.
        :return: True if the jump is feasible, False otherwise.
        """
        # Calculate horizontal distance using Euclidean distance in XY plane.
        horizontal_distance = math.sqrt((end[0] - start[0])**2 + (end[1] - start[1])**2)
        # Vertical difference (end.z - start.z)
        vertical_distance = end[2] - start[2]
        
        # Check horizontal and vertical constraints
        if horizontal_distance <= self.horizontal_range and vertical_distance <= self.jump_height:
            return True
        return False

    def can_reach(self, point, target):
        """
        Check if the robot's arm can reach a target from a given point.
        
        :param point: Tuple (x, y, z) representing the robot's position.
        :param target: Tuple (x, y, z) representing the target point.
        :return: True if the target is within reach, False otherwise.
        """
        distance = math.sqrt((target[0] - point[0])**2 +
                             (target[1] - point[1])**2 +
                             (target[2] - point[2])**2)
        return distance <= self.reach_range

    def move_to(self, new_position):
        """
        Update the robot's current position.
        
        :param new_position: Tuple (x, y, z) representing the new position.
        """
        self.position = new_position

    def visualize(self, ax, color='black'):
        """
        Visualize the robot's current position in a 3D plot.
        
        :param ax: Matplotlib 3D Axes object.
        :param color: Color for the robot's marker.
        """
        ax.scatter(*self.position, color=color, s=50, label="Robot")
        
    def __repr__(self):
        return (f"Robot(position={self.position}, jump_height={self.jump_height}, "
                f"horizontal_range={self.horizontal_range}, move_range={self.move_range}, "
                f"reach_range={self.reach_range})")


# ------------------------------------------
# 2. Block Class
# ------------------------------------------
class Block:
    def __init__(self, centroid, length, width, height, color="gray", name=None):
        """
        Initialize a movable block.

        :param centroid: Tuple (x, y, z) representing the bottom surface centroid.
        :param length: Length of the block (x-direction).
        :param width: Width of the block (y-direction).
        :param height: Height of the block (z-direction).
        :param color: Color used for visualization.
        """
        self.centroid = centroid  # Bottom surface centroid.
        self.length = length
        self.width = width
        self.height = height
        self.color = color
        self.name     = name
        self.bounding_box = self.compute_bounds()
        # Track the top centroid: bottom centroid plus height in z-direction.
        self.topcentroid = (centroid[0], centroid[1], centroid[2] + height)
        # Optionally keep track of previous positions for PRM updates.
        self.prev_centroid = None
        self.prev_topcentroid = None

    def compute_bounds(self):
        """
        Compute the 2D bounding box (as a Shapely Polygon) of the block's footprint.
        :return: Shapely Polygon representing the block's footprint.
        """
        x, y, _ = self.centroid
        half_length = self.length / 2
        half_width = self.width / 2
        return Polygon([
            (x - half_length, y - half_width),
            (x + half_length, y - half_width),
            (x + half_length, y + half_width),
            (x - half_length, y + half_width)
        ])

    def update_position(self, new_centroid):
        """
        Update the block's position and recompute its bounding box and top centroid.
        The previous bottom and top centroids are stored.
        
        :param new_centroid: Tuple (x, y, z) representing the new bottom surface centroid.
        """
        self.prev_centroid = self.centroid
        self.prev_topcentroid = self.topcentroid
        self.centroid = new_centroid
        self.topcentroid = (new_centroid[0], new_centroid[1], new_centroid[2] + self.height)
        self.bounding_box = self.compute_bounds()

    def visualize(self, ax):
        """
        Visualize the block as a cuboid in 3D.
        
        :param ax: Matplotlib 3D Axes object.
        """
        x, y, z = self.centroid
        half_length = self.length / 2
        half_width = self.width / 2
        
        # Define vertices for bottom and top faces.
        bottom = [
            (x - half_length, y - half_width, z),
            (x + half_length, y - half_width, z),
            (x + half_length, y + half_width, z),
            (x - half_length, y + half_width, z)
        ]
        top = [(vx, vy, z + self.height) for vx, vy, _ in bottom]
        
        # Create list of faces (each face is a list of vertices)
        faces = [
            bottom,       # bottom face
            top,          # top face
            [bottom[0], bottom[1], top[1], top[0]],  # side face 1
            [bottom[1], bottom[2], top[2], top[1]],  # side face 2
            [bottom[2], bottom[3], top[3], top[2]],  # side face 3
            [bottom[3], bottom[0], top[0], top[3]]   # side face 4
        ]
        
        # Plot the cuboid using Poly3DCollection.
        from mpl_toolkits.mplot3d.art3d import Poly3DCollection
        poly3d = Poly3DCollection(faces, facecolors=self.color, alpha=1, edgecolor='k')
        ax.add_collection3d(poly3d)

    def __repr__(self):
        return (f"Block(centroid={self.centroid}, topcentroid={self.topcentroid}, "
                f"dimensions=({self.length}, {self.width}, {self.height}), color={self.color})")


# ------------------------------------------
# 3. Plane Class
# ------------------------------------------

class Plane:
    def __init__(self, name, vertices):
        """
        Initialize a plane defined by its vertices.
        
        :param name: Name or identifier for the plane (e.g., "Floor", "Table").
        :param vertices: List of tuples (x, y, z) representing the vertices of the polygon.
        """
        self.name = name
        self.vertices = vertices
        # Create a 2D polygon (ignoring z) for the plane's footprint.
        self.shape = Polygon([(x, y) for x, y, z in vertices])
        # Obstacles on this plane are stored as a list of Shapely Polygons.
        self.obstacles = []

    def add_obstacle(self, obstacle_vertices):
        """
        Add an obstacle to the plane using its vertices.
        
        :param obstacle_vertices: List of tuples (x, y, z) representing the obstacle polygon.
        """
        obstacle = Polygon([(x, y) for x, y, z in obstacle_vertices])
        # Ensure the obstacle is within the plane's shape.
        if self.shape.contains(obstacle):
            self.obstacles.append(obstacle)
        else:
            raise ValueError("Obstacle is not fully contained within the plane.")

    def add_plane_as_obstacle(self, other_plane):
        """
        Project a higher plane onto this plane as an obstacle.
        
        :param other_plane: Another Plane object with a higher z-coordinate.
        """
        projected_shape = Polygon([(x, y) for x, y, z in other_plane.vertices])
        if self.shape.intersects(projected_shape):
            intersection = self.shape.intersection(projected_shape)
            self.obstacles.append(intersection)

    def is_point_free(self, point):
        """
        Check if a given point is in the free space of the plane (not in an obstacle).
        
        :param point: Tuple (x, y, z) representing the point.
        :return: True if the point is free, False otherwise.
        """
        p = Point(point[0], point[1])
        if not self.shape.contains(p):
            return False
        for obs in self.obstacles:
            if obs.contains(p):
                return False
        return True

    def sample_points(self, num_samples):
        """
        Generate random sample points from the free area of the plane.
        
        :param num_samples: Number of sample points to generate.
        :return: List of tuples (x, y, z) for free points.
        """
        samples = []
        minx, miny, maxx, maxy = self.shape.bounds
        count = 0
        while count < num_samples:
            x = random.uniform(minx, maxx)
            y = random.uniform(miny, maxy)
            # Use the z-coordinate from the first vertex (assumes constant z)
            z = self.vertices[0][2]
            if self.is_point_free((x, y, z)):
                samples.append((x, y, z))
                count += 1
        return samples

    def visualize(self, ax, plane_color="lightblue"):
        """
        Visualize the plane and its obstacles on a Matplotlib 3D Axes.
        
        :param ax: Matplotlib 3D Axes object.
        :param plane_color: Color for the plane.
        :return: The modified Axes object.
        """
        from mpl_toolkits.mplot3d.art3d import Poly3DCollection
        
        # Draw the plane polygon.
        poly = Poly3DCollection([self.vertices], alpha=0.5, facecolor=plane_color, edgecolor="k")
        ax.add_collection3d(poly)
        
        # Draw obstacles in red.
        for obs in self.obstacles:
            # Extract exterior coordinates.
            x, y = obs.exterior.xy
            # Use the same z as the plane.
            z = [self.vertices[0][2]] * len(x)
            ax.plot_trisurf(x, y, z, color="red", alpha=0.6)
        return ax

    def __repr__(self):
        return f"Plane(name={self.name}, vertices={self.vertices}, obstacles={len(self.obstacles)})"

def add_higher_plane_obstacles(planes):
    """
    Given a list of Plane objects, sort them by their z-levels (assumes constant z per plane)
    and add each higher plane as an obstacle to all lower planes.
    """
    # Sort planes by z-coordinate (assumed constant for each plane)
    sorted_planes = sorted(planes, key=lambda p: p.vertices[0][2])
    for i, lower_plane in enumerate(sorted_planes):
        for higher_plane in sorted_planes[i+1:]:
            lower_plane.add_plane_as_obstacle(higher_plane)
            


###############################################
# Helper Function
###############################################

def is_under_block(position, blocks, plane_z, tol=1e-6):
    """
    Check if a given position (x, y, z) is underneath any block.
    A position is considered under a block if:
      - The block's bottom centroid z is within tol of plane_z, and
      - The (x, y) of the position is contained in the block's bounding box.
    
    :param position: Tuple or array (x, y, z).
    :param blocks: List of Block objects.
    :param plane_z: The z value of the plane being considered.
    :param tol: Tolerance for comparing z values.
    :return: True if the position is under any block on this plane.
    """
    x, y, z = position
    pt = Point(x, y)
    for block in blocks:
        # Check if the block is on this plane.
        if abs(block.centroid[2] - plane_z) < tol:
            if block.bounding_box.contains(pt):
                return True
    return False

###############################################
# Local PRM Class
###############################################

class LocalPRM:
    def __init__(self, plane, connection_threshold, all_blocks):
        """
        Initialize a local PRM for a given plane.
        
        :param plane: A Plane object.
        :param connection_threshold: Maximum allowed distance between nodes on this plane.
        :param all_blocks: List of all Block objects in the environment.
                          (We use the block's bottom centroid z to decide if it's on this plane.)
        """
        self.plane = plane
        self.connection_threshold = connection_threshold
        # Vectorized storage of node positions (Nx3).
        self.nodes = np.empty((0, 3))
        self.metadata = []  # E.g., {"source": plane.name, "source_type": "Plane"}
        self.edges = []     # List of (i, j, weight)
        self.nx_graph = nx.Graph()
        self.all_blocks = all_blocks  # All blocks in the environment.

    def add_node(self, position, meta=None):
        """
        Add a node to the local PRM.
        """
        pos_arr = np.array(position).reshape(1, 3)
        self.nodes = np.vstack([self.nodes, pos_arr])
        index = self.nodes.shape[0] - 1
        if meta is None:
            meta = {"source": self.plane.name, "source_type": "Plane"}
        self.metadata.append(meta)
        self.nx_graph.add_node(index, position=position, meta=meta)
        return index

    def sample_nodes(self, num_samples):
        """
        Sample nodes using the plane's sampling function.
        (Nodes may lie under a block; connection methods later will skip those.)
        """
        samples = self.plane.sample_points(num_samples)
        for s in samples:
            self.add_node(s, meta={"source": self.plane.name, "source_type": "Plane", "blocks": self.all_blocks})
        return samples

    def connect_nodes(self):
        """
        Connect nodes in the local PRM using vectorized distance computation.
        Skips nodes that lie under a block (based on plane z).
        """
        n = self.nodes.shape[0]
        if n == 0:
            return
        dist_matrix = cdist(self.nodes, self.nodes)
        plane_z = self.plane.vertices[0][2]
        for i in range(n):
            if is_under_block(self.nodes[i], self.all_blocks, plane_z):
                continue
            for j in range(i+1, n):
                if is_under_block(self.nodes[j], self.all_blocks, plane_z):
                    continue
                if dist_matrix[i, j] <= self.connection_threshold:
                    weight = dist_matrix[i, j]
                    self.edges.append((i, j, weight))
                    self.nx_graph.add_edge(i, j, weight=weight)

    def update_node(self, index, new_position):
        """
        Update the position of a node (e.g., when a block moves).
        Uses vectorized computation to update edges for the node.
        """
        self.nodes[index] = np.array(new_position)
        # Remove edges incident to this node.
        self.edges = [edge for edge in self.edges if index not in edge[:2]]
        if self.nx_graph.has_node(index):
            for neighbor in list(self.nx_graph.neighbors(index)):
                self.nx_graph.remove_edge(index, neighbor)
        # Compute distances from new_position to all other nodes.
        dists = cdist(np.array([new_position]), self.nodes)[0]
        plane_z = self.plane.vertices[0][2]
        n = self.nodes.shape[0]
        for j in range(n):
            if j == index:
                continue
            # Skip if either node is under a block.
            if is_under_block(new_position, self.all_blocks, plane_z) or is_under_block(self.nodes[j], self.all_blocks, plane_z):
                continue
            if dists[j] <= self.connection_threshold:
                weight = dists[j]
                self.edges.append((min(index, j), max(index, j), weight))
                self.nx_graph.add_edge(index, j, weight=weight)

    def rebuild(self):
        """
        Rebuild the entire edge list and NetworkX graph.
        Uses vectorized distance computations.
        """
        self.edges = []
        self.nx_graph.clear()
        n = self.nodes.shape[0]
        # Add all nodes to the nx_graph.
        for i, pos in enumerate(self.nodes):
            self.nx_graph.add_node(i, position=tuple(pos), meta=self.metadata[i])
        if n == 0:
            return
        dist_matrix = cdist(self.nodes, self.nodes)
        plane_z = self.plane.vertices[0][2]
        for i in range(n):
            if is_under_block(self.nodes[i], self.all_blocks, plane_z):
                continue
            for j in range(i+1, n):
                if is_under_block(self.nodes[j], self.all_blocks, plane_z):
                    continue
                if dist_matrix[i, j] <= self.connection_threshold:
                    weight = dist_matrix[i, j]
                    self.edges.append((i, j, weight))
                    self.nx_graph.add_edge(i, j, weight=weight)

    def __repr__(self):
        return f"LocalPRM(plane={self.plane.name}, num_nodes={self.nodes.shape[0]}, num_edges={len(self.edges)})"

###############################################
# Global PRM Class
###############################################

class GlobalPRM:
    def __init__(self, jump_threshold, horizontal_range, block_extra_factor=2.0):
        """
        Initialize the global PRM.
        
        :param jump_threshold: Maximum allowed vertical jump distance.
        :param horizontal_range: Maximum horizontal jump distance for plane nodes.
        :param block_extra_factor: Multiplier for horizontal range when connecting block nodes.
        """
        self.nodes = np.empty((0, 3))
        self.metadata = []  # Metadata for each node.
        self.edges = []     # List of (i, j, weight)
        self.nx_graph = nx.Graph()
        self.jump_threshold = jump_threshold
        self.horizontal_range = horizontal_range
        self.block_extra_factor = block_extra_factor
        self.block_node_index = {}   # maps a stable block uid -> node index
        self.blocks = None           # authoritative block list for geometry tes

    def add_node(self, position, meta):
        pos_arr = np.array(position).reshape(1, 3)
        self.nodes = np.vstack([self.nodes, pos_arr])
        index = self.nodes.shape[0] - 1
        self.metadata.append(meta)
        self.nx_graph.add_node(index, position=position, meta=meta)
        return index
    
    def clone(self):
        g = GlobalPRM(self.jump_threshold, self.horizontal_range, self.block_extra_factor)
        # array copies
        g.nodes = self.nodes.copy()
        g.metadata = [m.copy() for m in self.metadata]
        g.edges = list(self.edges)  # shallow is fine: tuples of ints/floats
        # networkx graph copy (fast)
        g.nx_graph = self.nx_graph.copy(as_view=False)
        # keep block index map if present
        if hasattr(self, "block_node_index"):
            g.block_node_index = dict(self.block_node_index)
        return g
    
    def _block_index_for(self, block):
        # Prefer the per-object index if present (safe across deepcopy)
        if hasattr(block, "_gidx"):
            return block._gidx
        # Fallback via uid mapping
        uid = getattr(block, "uid", None)
        if uid is None:
            raise ValueError("Block has no uid/_gidx; did you call build_from_local?")
        return self.block_node_index[uid]

    # ---------- Edge book-keeping ----------
    def _remove_incident_edges(self, idx):
        """Remove all edges incident to node idx in both self.edges and self.nx_graph."""
        # strip from edge list
        self.edges = [e for e in self.edges if idx not in e[:2]]
        # strip from nx graph
        if self.nx_graph.has_node(idx):
            for nbr in list(self.nx_graph.neighbors(idx)):
                self.nx_graph.remove_edge(idx, nbr)
    
    def _add_edge_if_absent(self, i, j, weight):
        a, b = (i, j) if i < j else (j, i)
        # Append to list and nx (nx guards against duplicates; our list we keep clean by construction)
        self.edges.append((a, b, weight))
        self.nx_graph.add_edge(a, b, weight=weight)
    
    # ---------- Local target shortlisting (bbox) ----------
    def _candidate_targets_in_bbox(self, center_bbox, z_ok_mask=None, exclude_idx=None):
        """
        Return indices of nodes whose XY lie in bbox. Optionally AND with a z_ok_mask.
        Optionally exclude a node index.
        """
        xy = self.nodes[:, :2]
        mask = _points_in_bbox_mask(xy, center_bbox)
        if z_ok_mask is not None:
            mask &= z_ok_mask
        idxs = np.nonzero(mask)[0]
        if exclude_idx is not None:
            idxs = idxs[idxs != exclude_idx]
        return idxs
    
    # ---------- Geometry checks that can ignore the carried block ----------
    def _is_under_block_ignoring(self, pos, blocks_all, ignore_block, plane_z, tol=1e-6):
        """True if 'pos' is under any block in blocks_all except 'ignore_block' (same as is_under_block, but skip one)."""
        x, y = pos[:2]
        pt = Point(x, y)
        for blk in blocks_all:
            if blk is ignore_block:
                continue
            if abs(blk.centroid[2] - plane_z) < tol and blk.bounding_box.contains(pt):
                return True
        return False
    
    def _edge_intersects_block_ignoring(self, p1, p2, blocks_all, ignore_block):
        from shapely.geometry import LineString
        line = LineString([(p1[0], p1[1]), (p2[0], p2[1])])
        for blk in blocks_all:
            if blk is ignore_block:
                continue
            if line.intersects(blk.bounding_box):
                return True
        return False
    
    # ---------- Vectorized local re-connection for PLANE nodes ----------
    def _local_reconnect_plane_node(self, i, targets, ignore_block=None):
        """
        Recompute edges for a plane node i against a small 'targets' list.
        Applies the same gating as rebuild_edges() for plane↔plane edges, but:
          - can ignore one block (carried block),
          - uses only local target set (vectorized).
        """
        if targets.size == 0:
            return
        pos_i = self.nodes[i]
        meta_i = self.metadata[i]
        assert meta_i.get("source_type") == "Plane"
        blocks_all = meta_i.get("blocks", [])
        # Compute deltas
        pos_targets = self.nodes[targets]
        dxy = np.linalg.norm(pos_targets[:, :2] - pos_i[:2], axis=1)
        dz  = np.abs(pos_targets[:, 2] - pos_i[2])
        # Range gates
        ok = (dz <= self.jump_threshold) & (dxy <= self.horizontal_range)
        if not np.any(ok):
            return
        cand_ids = targets[ok]
        # Remove current incident edges first; we'll re-add only valid ones
        # (We only remove edges between i and candidates we might affect, keeping it cheap).
        for j in cand_ids:
            if self.nx_graph.has_edge(i, j):
                self.nx_graph.remove_edge(i, j)
        # Now add back edges that aren't invalidated by blocks
        for j in cand_ids:
            pos_j = self.nodes[j]
            # Skip if either endpoint lies under ANY block (except the carried block)
            under_i = self._is_under_block_ignoring(pos_i, blocks_all, ignore_block, plane_z=pos_i[2])
            under_j = self._is_under_block_ignoring(pos_j, blocks_all, ignore_block, plane_z=pos_j[2])
            if under_i or under_j:
                continue
            # Skip if segment crosses any block (except carried)
            if self._edge_intersects_block_ignoring(pos_i, pos_j, blocks_all, ignore_block):
                continue
            w = float(np.linalg.norm(pos_j - pos_i))
            self._add_edge_if_absent(i, j, w)
    
    # ---------- Vectorized local connect for BLOCK node ----------
    def _local_connect_block_node(self, bidx, targets, forbid_under_block):
        """
        Recompute edges between BLOCK node 'bidx' and local 'targets':
          - horizontal_range * block_extra_factor
          - |dz| <= jump_threshold
          - forbid connecting to plane nodes currently under the (placed) block (boolean mask)
        """
        if targets.size == 0:
            return
        pos_b  = self.nodes[bidx]
        pos_T  = self.nodes[targets]
        dxy    = np.linalg.norm(pos_T[:, :2] - pos_b[:2], axis=1)
        dz     = np.abs(pos_T[:, 2] - pos_b[2])
        ok     = (dz <= self.jump_threshold) & (dxy <= self.horizontal_range * self.block_extra_factor)
        if forbid_under_block is not None:
            ok &= ~forbid_under_block  # do not connect to nodes under the block
        cand_ids = targets[ok]
        # drop existing edges between bidx and these candidates, then re-add
        for j in cand_ids:
            if self.nx_graph.has_edge(bidx, j):
                self.nx_graph.remove_edge(bidx, j)
        for j in cand_ids:
            w = float(np.linalg.norm(self.nodes[j] - pos_b))
            self._add_edge_if_absent(bidx, j, w)
    
    # ==========================================================
    #                PUBLIC:  fast Pick / Place
    # ==========================================================
    def fast_on_pick(self, block, margin=None):
        """
        Called when 'block' is picked up.
        Steps:
          1) Remove ALL edges incident to the block's top node.
          2) Reconnect ONLY plane nodes inside the old footprint +/- margin,
             to nearby nodes (bbox shortlist) while IGNORING this carried block
             in under-block and intersection tests.
        """
        if margin is None:
            margin = float(self.horizontal_range)
        # 1) remove edges of the block’s top node
        bidx = self._block_index_for(block)
        self._remove_incident_edges(bidx)
    
        # 2) gather candidate plane nodes inside (expanded) old footprint
        old_bbox = _bbox_expand(block.bounding_box.bounds, margin)
        xy = self.nodes[:, :2]; z = self.nodes[:, 2]
        bbox_mask = _points_in_bbox_mask(xy, old_bbox)
        # same-Z-as-plane mask: we want plane nodes (source_type == "Plane") near that z
        plane_mask = np.array([m.get("source_type") == "Plane" for m in self.metadata])
        # shortlist of plane nodes to refresh
        refresh_idxs = np.nonzero(bbox_mask & plane_mask)[0]
    
        if refresh_idxs.size == 0:
            return  # nothing to do
    
        # For each such node i, reconnect it to *local* targets via bbox shortlist, vectorized
        for i in refresh_idxs:
            # Candidates to test against: plane nodes in a slightly larger bbox around node i
            local_bbox = _bbox_expand((xy[i,0], xy[i,0], xy[i,1], xy[i,1]), margin)  # degenerate; expand works the same
            # Better: reuse the same old_bbox to reconnect in the same locality
            targets = self._candidate_targets_in_bbox(old_bbox, z_ok_mask=None, exclude_idx=i)
            if targets.size == 0:
                continue
            # But we only want plane nodes as targets (plane↔plane reconnect)
            targets = targets[ np.array([self.metadata[j].get("source_type")=="Plane" for j in targets]) ]
            if targets.size == 0:
                continue
            self._local_reconnect_plane_node(i, targets, ignore_block=block)
    
    def fast_on_place(self, block, new_centroid, margin=None, bbox_factor=2.0):
        """
        Called when 'block' is placed at 'new_centroid' (bottom).
        Steps:
          1) Identify plane nodes under NEW footprint (same plane z); delete *their* edges.
          2) Move the block top node to its new top centroid; delete its edges; reconnect
             to nearby nodes shortlisted by a bbox of size ~ (bbox_factor * horizontal_range).
          3) Do NOT connect the block node to plane nodes that lie under the block.
        """
        if margin is None:
            margin = float(self.horizontal_range)
    
        # Keep old bbox only if you also want to lightly refresh there; per your spec we work on NEW area
        # 0) Update the block's geometry
        block.update_position(new_centroid)
        new_top = block.topcentroid
        new_bbox = block.bounding_box.bounds
    
        # 1) remove edges from all plane nodes under the new footprint
        xy = self.nodes[:, :2]; z = self.nodes[:, 2]
        plane_mask = np.array([m.get("source_type") == "Plane" for m in self.metadata])
        # Plane Z must match the block's bottom Z (i.e., the plane where it sits)
        z_same = _z_equal_mask(z, new_centroid[2])
        under_mask = np.zeros(len(self.nodes), dtype=bool)
        # under test via polygon contains
        minx, miny, maxx, maxy = new_bbox
        mask_bbox = _points_in_bbox_mask(xy, new_bbox)
        cand_idxs = np.nonzero(mask_bbox & plane_mask & z_same)[0]
        # precise under check
        for i in cand_idxs:
            if block.bounding_box.contains(Point(xy[i,0], xy[i,1])):
                under_mask[i] = True
        under_idxs = np.nonzero(under_mask)[0]
        for i in under_idxs:
            self._remove_incident_edges(i)
    
        # 2) move + reconnect the block node
        bidx = self._block_index_for(block)
        # move node
        self.nodes[bidx] = np.array(new_top)
        # drop any leftovers
        self._remove_incident_edges(bidx)
    
        # shortlist targets around the new footprint (wider bbox)
        expand = bbox_factor * self.horizontal_range
        wide_bbox = _bbox_expand(new_bbox, expand)
        targets = self._candidate_targets_in_bbox(wide_bbox, exclude_idx=bidx)
    
        if targets.size == 0:
            return
    
        # forbid connecting block→plane to nodes that lie under the block (same vector we computed)
        forbid = None
        if targets.size:
            forbid = np.zeros(targets.size, dtype=bool)
            # mark forbidden where target is a plane node at same z and inside polygon
            for t_k, j in enumerate(targets):
                mj = self.metadata[j]
                if mj.get("source_type") != "Plane": 
                    continue
                if abs(self.nodes[j][2] - new_centroid[2]) > 1e-6:
                    continue
                if block.bounding_box.contains(Point(self.nodes[j][0], self.nodes[j][1])):
                    forbid[t_k] = True
    
        # Connect with block rules (range * block_extra_factor, |dz| <= jump)
        self._local_connect_block_node(bidx, targets, forbid_under_block=forbid)
        
        
        
    def rebuild_edges(self):
        """
        Rebuild the entire global edge list and update the NetworkX graph.
        Uses vectorized operations to precompute distance, vertical, and horizontal differences.
        Now also filters out plane nodes that are under a block.
        """
        self.edges = []
        self.nx_graph.clear()
        n = self.nodes.shape[0]
        for i in range(n):
            self.nx_graph.add_node(i, position=tuple(self.nodes[i]), meta=self.metadata[i])
        if n == 0:
            return
        dist_matrix = cdist(self.nodes, self.nodes)
        # Compute vertical differences and horizontal differences in a vectorized manner.
        z_coords = self.nodes[:, 2]
        vertical_diff = np.abs(z_coords.reshape(-1, 1) - z_coords.reshape(1, -1))
        horizontal_diff = np.linalg.norm(self.nodes[:, :2].reshape(n, 1, 2) - self.nodes[:, :2].reshape(1, n, 2), axis=2)
        for i in range(n):
            for j in range(i+1, n):
                meta_i = self.metadata[i]
                meta_j = self.metadata[j]
                pos_i = self.nodes[i]
                pos_j = self.nodes[j]
                # If both nodes come from a plane, check if either is under a block.
                if meta_i.get("source_type") == "Plane" and meta_j.get("source_type") == "Plane":
                    blocks_i = meta_i.get("blocks", [])
                    blocks_j = meta_j.get("blocks", [])
                    combined_blocks = blocks_i if blocks_i else blocks_j
                    # Skip edge if either node is under a block.
                    if is_under_block(pos_i, combined_blocks, pos_i[2]) or is_under_block(pos_j, combined_blocks, pos_j[2]):
                        continue
                    if edge_intersects_block(pos_i, pos_j, combined_blocks):
                        continue
                    # if is_under_block(pos_i, blocks_i, pos_i[2]) or is_under_block(pos_j, blocks_j, pos_j[2]):
                    #     continue  # Skip connecting these nodes.
                    # Otherwise, connect if within jump and horizontal range.
                    if vertical_diff[i, j] <= self.jump_threshold and horizontal_diff[i, j] <= self.horizontal_range:
                        weight = dist_matrix[i, j]
                        self.edges.append((i, j, weight))
                        self.nx_graph.add_edge(i, j, weight=weight)
                elif meta_i.get("source_type") == "Block" or meta_j.get("source_type") == "Block":
                    # For block nodes, use extended horizontal range.
                    if vertical_diff[i, j] <= self.jump_threshold and horizontal_diff[i, j] <= self.horizontal_range * self.block_extra_factor:
                        weight = dist_matrix[i, j]
                        self.edges.append((i, j, weight))
                        self.nx_graph.add_edge(i, j, weight=weight)
                else:
                    if vertical_diff[i, j] <= self.jump_threshold and horizontal_diff[i, j] <= self.horizontal_range:
                        weight = dist_matrix[i, j]
                        self.edges.append((i, j, weight))
                        self.nx_graph.add_edge(i, j, weight=weight)
    
    
    def build_from_local(self, local_prms, blocks):
        """
        Build the global PRM by:
          - Concatenating nodes from all local PRMs.
          - Adding block top centroid nodes.
          - Rebuilding connectivity.
          
        :param local_prms: List of LocalPRM objects.
        :param blocks: List of Block objects.
        """
        # Reset global data.
        self.nodes = np.empty((0, 3))
        self.metadata = []
        self.edges = []
        self.nx_graph.clear()
        self.block_node_index = {}
        self.blocks = blocks  # keep reference for geometry checks
        
        # add plane nodes
        for local in local_prms:
            for i in range(local.nodes.shape[0]):
                pos = local.nodes[i]
                meta = local.metadata[i].copy()
                meta["source_type"] = "Plane"
                # use self.blocks instead of storing blocks in every node’s meta
                self.add_node(pos, meta)
        
        # add one node per block (top centroid)
        for k, blk in enumerate(blocks):
            # give each block a stable uid once (persists across clones and deepcopies)
            if not hasattr(blk, "uid"):
                blk.uid = k  # or any stable integer you prefer
            idx = self.add_node(blk.topcentroid, {"source_type": "Block", "block_uid": blk.uid})
            self.block_node_index[blk.uid] = idx
            # store the global node index on the block object too (copied across deepcopies)
            blk._gidx = idx
        
        self.rebuild_edges()

    def __repr__(self):
        return (f"GlobalPRM(num_nodes={self.nodes.shape[0]}, num_edges={len(self.edges)}, "
                f"jump_threshold={self.jump_threshold}, horizontal_range={self.horizontal_range})")
    


###############################################
# Helper Function
###############################################
def find_valid_gaps(planes, dgap):
    """
    Given a list of Plane objects, return a list of index pairs (i, j) that represent valid gaps.
    A valid gap exists between two planes if the minimum horizontal distance between their 2D shapes
    (i.e., the distance computed by the Shapely Polygon's distance function) is less than dgap.
    
    Parameters:
      planes: List of Plane objects.
      dgap: User-defined maximum gap distance for a gap to be considered valid.
    
    Returns:
      A list of tuples (i, j). Each tuple represents a directional valid gap.
      For example, if (0, 1) is returned, it means plane 0 and plane 1 are close enough.
      (Both directions are returned: (0, 1) and (1, 0).)
    """
    valid_pairs = []
    n = len(planes)
    for i in range(n):
        for j in range(i + 1, n):
            gap = planes[i].shape.distance(planes[j].shape)
            if gap < dgap:
                valid_pairs.append((i, j))
                # valid_pairs.append((j, i))
    return valid_pairs

def candidate_blocks_for_gaps(valid_gaps, blocks, planes, robot):
    """
    Candidate(Pi,Pj): lightweight geometric filter (conservative superset).

    We include (block_n, Pk) for gap (Pi,Pj) if:
      (1) Height (δH): top_z(Pk, block_n) is within robot.jump_height of BOTH Pi and Pj
      (2) Horizontal (δL): Pk is within robot.horizontal_range (in XY) of BOTH Pi and Pj
          (using min distance between plane footprints in XY).

    Returns:
      dict[(i,j)] -> list of (block_idx, candidate_plane_obj)
    """

    # δH and δL from the robot object
    delta_H = float(getattr(robot, "jump_height", 0.0))
    delta_L = float(getattr(robot, "horizontal_range", float("inf")))

    def plane_z(pl):
        # assumes horizontal planes (your codebase assumption)
        return float(pl.vertices[0][2])

    def xy_plane_distance(pA, pB):
        # Prefer shapely polygon distance if available
        if hasattr(pA, "shape") and pA.shape is not None and hasattr(pB, "shape") and pB.shape is not None:
            try:
                return float(pA.shape.distance(pB.shape))
            except Exception:
                pass

        # Fallback (no shapely): min distance between vertex XY sets (conservative-ish)
        import math
        min_d = float("inf")
        for (ax, ay, _az) in pA.vertices:
            for (bx, by, _bz) in pB.vertices:
                d = math.hypot(ax - bx, ay - by)
                if d < min_d:
                    min_d = d
        return float(min_d)

    gap_candidates = {}

    for gap in valid_gaps:
        i, j = gap
        Pi = planes[i]
        Pj = planes[j]
        z_i = plane_z(Pi)
        z_j = plane_z(Pj)

        candidates = []

        for Pk in planes:
            # δL: Pk must be "near enough" (in XY) to both Pi and Pj
            if xy_plane_distance(Pk, Pi) > delta_L:
                continue
            if xy_plane_distance(Pk, Pj) > delta_L:
                continue

            pk_z = plane_z(Pk)

            for b_idx, block in enumerate(blocks):
                # δH: top surface height after placing on Pk
                top_z = pk_z + float(block.height)

                # Must be within step height of BOTH planes (up or down)
                if abs(top_z - z_i) <= delta_H and abs(top_z - z_j) <= delta_H:
                    candidates.append((b_idx, Pk))

        gap_candidates[gap] = candidates

    return gap_candidates

# def candidate_blocks_for_gaps(valid_gaps, blocks, planes, robot):
#     """
#     For each valid gap (a pair of plane indices), find all candidate (block, placement_plane)
#     combinations that can bridge the gap. A candidate is valid if, when the block is placed
#     on a candidate plane, its top surface is within the robot's jump capability from both planes.
    
#     We assume that each Plane object's z coordinate is given by its first vertex.
    
#     :param valid_gaps: List of tuples (i, j) of indices into the 'planes' list that represent valid gaps.
#     :param blocks: List of Block objects.
#     :param planes: List of Plane objects.
#     :param robot: A Robot object (must have a 'jump_height' attribute).
#     :return: A dictionary mapping each gap (tuple (i, j)) to a list of candidate tuples (block, candidate_plane).
#     """
#     gap_candidates = {}
    
#     for gap in valid_gaps:
#         i, j = gap
#         # Determine lower and upper planes based on their z-coordinate.
#         z_i = planes[i].vertices[0][2]
#         z_j = planes[j].vertices[0][2]
#         z_lower = min(z_i, z_j)
#         z_upper = max(z_i, z_j)
        
#         candidates = []
#         # For every candidate placement plane and every block,
#         # check if placing the block on that plane yields a top surface that bridges the gap.
#         for candidate_plane in planes:
#             # Get the plane's z (assuming constant z for all vertices)
#             plane_z = candidate_plane.vertices[0][2]
#             for idx, block in enumerate(blocks):
#                 # When placed on this candidate plane, the block's top is:
#                 candidate_top = plane_z + block.height
#                 # Use absolute differences because the robot can jump up or down.
#                 if abs(candidate_top - z_lower) <= robot.jump_height and abs(z_upper - candidate_top) <= robot.jump_height:
#                     candidates.append((idx, candidate_plane))
#         gap_candidates[gap] = candidates

#     return gap_candidates
def make_candidate_dict_from_user_names(user_candidates, planes, blocks):
    """
    Convert user-friendly candidate mapping into the planner-internal structure.

    Parameters
    ----------
    user_candidates : dict
        Keys are gap names: either "PlaneA-PlaneB" or (PlaneA, PlaneB).
        Values are lists of (block_name, place_plane_name).
    planes : list[Plane]
        The Plane objects; each must have a unique .name.
    blocks : list[Block]
        The Block objects; each must have a unique .name.

    Returns
    -------
    dict[tuple[int,int], list[tuple[int, Plane]]]
        Keys are (i,j) with i<j (indices into `planes`).
        Values are (block_index, placement_plane_object) tuples.
    """
    # --- lookups ---
    plane_idx = {p.name: i for i, p in enumerate(planes)}
    if len(plane_idx) != len(planes):
        raise ValueError("Duplicate plane names detected.")

    plane_obj = {p.name: p for p in planes}

    block_idx = {}
    for i, b in enumerate(blocks):
        if not hasattr(b, "name") or b.name is None:
            raise ValueError(f"Block at index {i} has no .name; set Block(..., name='B{i+1}')")
        if b.name in block_idx:
            raise ValueError(f"Duplicate block name '{b.name}'.")
        block_idx[b.name] = i

    # --- convert ---
    out = {}
    for gap_key, cand_list in user_candidates.items():
        # Parse gap key
        if isinstance(gap_key, str):
            parts = [x.strip() for x in gap_key.split("-")]
            if len(parts) != 2:
                raise ValueError(f"Gap key '{gap_key}' must be 'PlaneA-PlaneB'.")
            g1, g2 = parts
        elif isinstance(gap_key, (tuple, list)) and len(gap_key) == 2:
            g1, g2 = gap_key
        else:
            raise TypeError("Gap key must be 'A-B' string or a (A,B) tuple/list.")

        try:
            i = plane_idx[g1]; j = plane_idx[g2]
        except KeyError as e:
            raise KeyError(f"Unknown plane name in gap '{gap_key}': {e.args[0]}")

        ij = (i, j) if i < j else (j, i)

        # Map each (block_name, placement_plane_name)
        converted = []
        for blk_name, place_plane_name in cand_list:
            try:
                bi = block_idx[blk_name]
            except KeyError:
                raise KeyError(f"Unknown block name '{blk_name}' in candidates for gap {gap_key}.")
            try:
                place_plane = plane_obj[place_plane_name]
            except KeyError:
                raise KeyError(f"Unknown placement plane '{place_plane_name}' in candidates for gap {gap_key}.")
            converted.append((bi, place_plane))

        # Merge if user specified both "A-B" and "B-A"
        if ij not in out:
            out[ij] = converted
        else:
            out[ij].extend(converted)

    return out

def distance_to_rectangle_boundary(point, bounds):
    """
    Compute the distance from a point (x, y) to the boundary of an axis-aligned rectangle.
    If the point is inside the rectangle, the distance is the minimum distance from the point 
    to any of the four sides.
    
    :param point: Tuple or array (x, y).
    :param bounds: Tuple (minx, miny, maxx, maxy).
    :return: Distance from the point to the rectangle's boundary.
    """
    x, y = point
    minx, miny, maxx, maxy = bounds
    # If the point is inside, compute the minimal distance to any side.
    if minx <= x <= maxx and miny <= y <= maxy:
        return min(x - minx, maxx - x, y - miny, maxy - y)
    else:
        # If outside, compute Euclidean distance to the rectangle.
        dx = max(minx - x, 0, x - maxx)
        dy = max(miny - y, 0, y - maxy)
        return np.hypot(dx, dy)

def sample_block_placement_general(gap, candidate_plane, block, robot, planes, blocks, current_prm=None, max_attempts=100):
    """
    Unified placement sampler with OPTIONAL early bridge-intent verification.
      - If gap is a tuple (i, j): we repeatedly propose placements and, if a PRM is
        provided via current_prm, simulate {fast_on_pick → fast_on_place} on a TEMP clone
        and require that Pi↔Pj become connected in the simulated PRM before accepting.
      - If gap is None: sample a valid free placement anywhere on candidate_plane (no bridge check).

    Returns:
      (x, y, z) for the block's new bottom centroid on candidate_plane, or None.
    """
    # Local helper: pick a random, live, not-under-block node on a named plane
    def _random_plane_node_idx(prm, plane_name, blocks, require_connected=True, max_tries=64):
        cand = [
            i for i, m in enumerate(prm.metadata)
            if m.get("source_type") == "Plane" and m.get("source") == plane_name
        ]
        if not cand:
            return None
        if not require_connected:
            return random.choice(cand)

        live = []
        for i in cand:
            if prm.nx_graph.degree(i) <= 0:
                continue
            pos = prm.nodes[i]
            if is_under_block(pos, blocks, pos[2]):
                continue
            live.append(i)
        if not live:
            return None

        for _ in range(max_tries):
            i = random.choice(live)
            if prm.nx_graph.has_node(i) and prm.nx_graph.degree(i) > 0:
                return i
        return None

    # ---- Case A: gap-guided with early bridge check if PRM is given ----
    if isinstance(gap, tuple) and len(gap) == 2:
        i, j = gap
        plane1 = planes[i]
        plane2 = planes[j]

        for _ in range(max_attempts):
            # Propose a candidate placement (one-at-a-time sampling)
            pt = sample_block_placement(plane1, plane2, candidate_plane, block, robot, max_attempts=1)
            if pt is None:
                continue

            # If no PRM was provided, we can't verify—fall back to accepting the proposal.
            if current_prm is None:
                return pt

            # Simulate on a TEMP clone: pick → place, then check Pi<->Pj connectivity.
            temp = current_prm.clone()
            orig_centroid = block.centroid
            try:
                temp.fast_on_pick(block)
                temp.fast_on_place(block, pt)

                n1 = _random_plane_node_idx(temp, plane1.name, blocks, require_connected=True)
                n2 = _random_plane_node_idx(temp, plane2.name, blocks, require_connected=True)
                if n1 is None or n2 is None:
                    # couldn't get usable nodes; try another placement sample
                    continue

                try:
                    _ = nx.astar_path(
                        temp.nx_graph, n1, n2,
                        heuristic=lambda a, b: np.linalg.norm(temp.nodes[a] - temp.nodes[b])
                    )
                    # bridged ✅
                    return pt
                except nx.NetworkXNoPath:
                    # not bridged → try another placement sample
                    pass
            finally:
                # Restore the caller's block object to its original centroid.
                try:
                    block.update_position(orig_centroid)
                except Exception:
                    pass

        # Exhausted attempts → report failure to the caller.
        return None

    # ---- Case B: 'None' intent → free placement on candidate_plane (no bridge check) ----
    z_B = candidate_plane.vertices[0][2]
    for _ in range(max_attempts):
        # Try the plane's own sampler first (uniform-ish over free space)
        pts = candidate_plane.sample_points(1)  # returns list
        if pts:
            candidate = pts[0]
        else:
            # Fallback: rejection sampling within plane bounds + is_point_free
            minx, miny, maxx, maxy = candidate_plane.shape.bounds
            candidate = (random.uniform(minx, maxx), random.uniform(miny, maxy), z_B)

        # Must be inside plane free space
        if not candidate_plane.is_point_free(candidate):
            continue

        # Also avoid dropping where another block already occupies (same plane z)
        if is_under_block(candidate, blocks, z_B):
            continue

        return candidate

    return None



def sample_block_placement(plane1, plane2, plane_B, block, robot, max_attempts=50):
    """
    Sample a valid block placement location (i.e. the block's new bottom centroid) 
    given two planes. This function:
      1. Determines the higher (Plane H) and lower (Plane L) among the two provided.
      2. Samples a candidate point along the boundary of Plane H.
      3. Offsets the sampled point outward by half the appropriate block dimension plus 0.2.
      4. Projects the candidate onto the candidate placement plane (Plane B) by setting its z
         to plane_B's z-coordinate.
      5. Checks that the candidate lies inside Plane B.
      6. If Plane B is not the same as Plane L, computes the horizontal distance from the candidate 
         to the boundary of Plane L and requires that distance to be within robot.horizontal_range.
      7. Performs collision checking using plane_B.is_point_free.
    
    If no valid candidate is found within max_attempts, returns None.
    
    :param plane1: First Plane object.
    :param plane2: Second Plane object.
    :param plane_B: Candidate placement Plane object.
    :param block: Block object with attributes length, width, height.
    :param robot: Robot object with attributes horizontal_range and jump_height.
    :param max_attempts: Maximum sampling attempts.
    :return: Candidate block bottom centroid (x, y, z) on plane_B if valid; else None.
    """
    # Determine Plane H (higher) and Plane L (lower) based on z-coordinate of first vertex.
    z1 = plane1.vertices[0][2]
    z2 = plane2.vertices[0][2]
    if z1 >= z2:
        plane_H = plane1
        plane_L = plane2
    else:
        plane_H = plane2
        plane_L = plane1

    # Get Plane H's bounding box (axis-aligned rectangle).
    minx, miny, maxx, maxy = plane_H.shape.bounds
    
    # Get Plane B's z coordinate.
    z_B = plane_B.vertices[0][2]
    # Get Plane L's bounding box for horizontal distance computation.
    L_bounds = plane_L.shape.bounds

    for attempt in range(max_attempts):
        # Randomly choose one of the four edges of Plane H.
        edge_choice = random.choice(['top', 'bottom', 'left', 'right'])
        
        if edge_choice == 'top':
            # Top edge: y = maxy, sample x uniformly.
            x_candidate = random.uniform(minx, maxx)
            y_candidate = maxy
            # Outward direction is upward (+y). Offset = block.width/2 + 0.2.
            offset = block.width / 2.0 + 0.2
            candidate_xy = np.array([x_candidate, y_candidate]) + np.array([0, offset])
        elif edge_choice == 'bottom':
            # Bottom edge: y = miny, sample x uniformly.
            x_candidate = random.uniform(minx, maxx)
            y_candidate = miny
            # Outward direction is downward (-y). Offset = block.width/2 + 0.2.
            offset = block.width / 2.0 + 0.2
            candidate_xy = np.array([x_candidate, y_candidate]) + np.array([0, -offset])
        elif edge_choice == 'left':
            # Left edge: x = minx, sample y uniformly.
            y_candidate = random.uniform(miny, maxy)
            x_candidate = minx
            # Outward direction is left (-x). Offset = block.length/2 + 0.2.
            offset = block.length / 2.0 + 0.2
            candidate_xy = np.array([x_candidate, y_candidate]) + np.array([-offset, 0])
        else:  # 'right'
            # Right edge: x = maxx, sample y uniformly.
            y_candidate = random.uniform(miny, maxy)
            x_candidate = maxx
            # Outward direction is right (+x). Offset = block.length/2 + 0.2.
            offset = block.length / 2.0 + 0.2
            candidate_xy = np.array([x_candidate, y_candidate]) + np.array([offset, 0])
        
        # Project candidate onto Plane B by setting its z to plane_B's z.
        candidate = np.array([candidate_xy[0], candidate_xy[1], z_B])
        
        # Immediately check if candidate (its XY) lies inside Plane B.
        if not plane_B.shape.contains(Point(candidate_xy)):
            continue
        
        # Collision check on Plane B.
        if not plane_B.is_point_free(candidate):
            continue
        
        # If Plane B is not the same as Plane L, check candidate's horizontal distance to the boundary of Plane L.
        if plane_B != plane_L:
            dist_to_boundary = distance_to_rectangle_boundary(candidate_xy, L_bounds)
            if dist_to_boundary > robot.horizontal_range:
                continue
        
        # If candidate passes all checks, return it.
        return tuple(candidate)
    
    # If no valid candidate was found, return None.
    return None 

def set_axes_equal(ax):
    '''Make axes of 3D plot have equal scale so that spheres appear as spheres,
    cubes as cubes, etc..  This is one possible solution to Matplotlib's ax.set_aspect('equal')
    and works for 3D plots.
    '''
    x_limits = ax.get_xlim3d()
    y_limits = ax.get_ylim3d()
    z_limits = ax.get_zlim3d()
    x_range = abs(x_limits[1] - x_limits[0])
    x_middle = np.mean(x_limits)
    y_range = abs(y_limits[1] - y_limits[0])
    y_middle = np.mean(y_limits)
    z_range = abs(z_limits[1] - z_limits[0])
    z_middle = np.mean(z_limits)
    plot_radius = 0.5 * max([x_range, y_range, z_range])
    ax.set_xlim3d([x_middle - plot_radius, x_middle + plot_radius])
    ax.set_ylim3d([y_middle - plot_radius, y_middle + plot_radius])
    ax.set_zlim3d([z_middle - plot_radius, z_middle + plot_radius])

def visualize_global_prm(global_prm, planes, block_list, title="Environment"):
    fig = plt.figure(figsize=(10, 8), dpi=250)
    ax = fig.add_subplot(111, projection='3d')
    # Plot each plane.
    plane_colors = {"Ground": "white", "Table1": "lightyellow", "Table2": "lightyellow"}
    for plane in planes:
        if plane.name != "Ground":
            poly = Poly3DCollection([plane.vertices], alpha=0.5, facecolor=plane_colors.get(plane.name, "lightyellow"))
            ax.add_collection3d(poly)
            xs = [v[0] for v in plane.vertices] + [plane.vertices[0][0]]
            ys = [v[1] for v in plane.vertices] + [plane.vertices[0][1]]
            zs = [v[2] for v in plane.vertices] + [plane.vertices[0][2]]
            ax.plot(xs, ys, zs, color="black")
    
    # Plot all blocks.
    for block in block_list:
        block.visualize(ax)
    
    # # Plot global PRM nodes.
    # ax.scatter(global_prm.nodes[:, 0], global_prm.nodes[:, 1], global_prm.nodes[:, 2],
    #            color="red", s=10, label="PRM Nodes")
    
    # Plot global PRM edges.
    count=0
    for edge in global_prm.edges:
        count+=1
        if count%1==0:
            i, j, weight = edge
            p1 = global_prm.nodes[i]
            p2 = global_prm.nodes[j]
            ax.plot([p1[0], p2[0]], [p1[1], p2[1]], [p1[2], p2[2]], color="magenta", linewidth=0.5, alpha=0.3)
    
    # ax.set_title(title)
    ax.set_xlabel("X")
    ax.set_ylabel("Y")
    ax.set_zlabel("Z")
    ax.set_xlim3d(-5, 32) 
    ax.set_zlim3d(0, 5) 
    ax.set_ylim3d(-2, 32) 
    # set_axes_equal(ax)
    # ax.set_box_aspect((1, 1, 0.4))
    plt.legend()
    plt.show()





def get_reachable_planes_from_robot(global_prm, robot_node_index):
    """
    Return a set of plane names that are reachable from the robot's current node
    in the global PRM.
    """
    if robot_node_index is None:
        return set()

    # Safety: if node index somehow not in graph
    if robot_node_index not in global_prm.nx_graph:
        return set()

    reachable_nodes = nx.node_connected_component(global_prm.nx_graph, robot_node_index)

    reachable_planes = set()
    for idx in reachable_nodes:
        meta = global_prm.metadata[idx]
        if meta.get("source_type") == "Plane":
            plane_name = meta.get("source")
            if plane_name is not None:
                reachable_planes.add(plane_name)

    return reachable_planes

def visualize_global_prm_reachable(global_prm,
                                   planes,
                                   block_list,
                                   robot_node_index=None,
                                   highlight_reachable=True,
                                   title="Environment"):
    """
    3D PRM visualizer (same style as visualize_global_prm), but with
    reachable planes highlighted based on the current robot node.
    """

    fig = plt.figure(figsize=(10, 8), dpi=250)
    ax = fig.add_subplot(111, projection='3d')

    # ----------------------------
    # Compute reachable planes
    # ----------------------------
    reachable_plane_names = set()
    if highlight_reachable and robot_node_index is not None:
        reachable_plane_names = get_reachable_planes_from_robot(global_prm, robot_node_index)

    # ----------------------------
    # Plot planes (3D polygons)
    # ----------------------------
    base_plane_colors = {"Ground": "white", "Table1": "lightyellow", "Table2": "lightyellow"}

    plane_colors = {"Ground": "white", "Table1": "lightyellow", "Table2": "lightyellow"}

    for plane in planes:
        xs = [v[0] for v in plane.vertices] + [plane.vertices[0][0]]
        ys = [v[1] for v in plane.vertices] + [plane.vertices[0][1]]
        zs = [v[2] for v in plane.vertices] + [plane.vertices[0][2]]
    
        # --- default appearance ---
        facecolor = plane_colors.get(plane.name, "lightyellow")
        edgecolor = "black"
        alpha = 0.5
    
        # --- highlight reachable planes (non-ground) ---
        if plane.name in reachable_plane_names and plane.name != "Ground":
            facecolor = "lightgreen"
            edgecolor = "green"
            alpha = 0.7
    
        if plane.name == "Ground":
            # NO FILL for ground: only outline
            # If you want a “reachable” hint for ground, you can bump the color:
            if plane.name in reachable_plane_names:
                edgecolor = "green"
            ax.plot(xs, ys, zs, color=edgecolor, linewidth=1.0)
        else:
            # Normal planes: filled patch + outline
            poly = Poly3DCollection([plane.vertices], alpha=alpha,
                                    facecolor=facecolor)
            ax.add_collection3d(poly)
            ax.plot(xs, ys, zs, color=edgecolor, linewidth=1.0)



    # ----------------------------
    # Plot all blocks (unchanged)
    # ----------------------------
    for block in block_list:
        block.visualize(ax)

    # ----------------------------
    # Plot global PRM edges (unchanged)
    # ----------------------------
    count = 0
    for edge in global_prm.edges:
        count += 1
        if count % 1 == 0:
            i, j, weight = edge
            p1 = global_prm.nodes[i]
            p2 = global_prm.nodes[j]
            ax.plot([p1[0], p2[0]], [p1[1], p2[1]], [p1[2], p2[2]],
                    color="magenta", linewidth=0.5, alpha=0.3)

    # ----------------------------
    # Optional: plot robot position
    # ----------------------------
    if robot_node_index is not None:
        rx, ry, rz = global_prm.nodes[robot_node_index]
        ax.scatter([rx], [ry], [rz], color="black", s=40, label="Robot")

    # ----------------------------
    # Axes formatting (same as old)
    # ----------------------------
    ax.set_xlabel("X")
    ax.set_ylabel("Y")
    ax.set_zlabel("Z")
    ax.set_xlim3d(-5, 32)
    ax.set_zlim3d(0, 5)
    ax.set_ylim3d(-2, 32)
    if title:
        ax.set_title(title)

    # Legend might be empty if no labeled artists; harmless
    try:
        plt.legend()
    except Exception:
        pass

    plt.tight_layout()
    plt.show()




class TreeNode:
    def __init__(self, robot_node_index, global_prm, block_config, actions, reach_hist=None, parent=None):
        """
        Represents a node in the search tree.
        
        :param robot_node_index: The index of the robot node in the global PRM.
        :param global_prm: A deep copy of the global PRM at this state.
        :param block_config: List of Block objects (their current configuration).
        :param actions: List of high-level actions taken so far.
        :param parent: Parent TreeNode (or None for root).
        """
        self.robot_node_index = robot_node_index
        self.global_prm = global_prm
        self.block_config = block_config  # For our purposes, we store the list (they are mutable objects).
        self.actions = actions
        self.reach_hist = [] if reach_hist is None else reach_hist
        self.parent = parent


class Planner:
    def __init__(self, global_prm, local_prms, blocks, robot, goal, goal_threshold, valid_gaps, planes,permanent_edges, user_candidates=None):
        """
        Initialize the planner.
        
        :param global_prm: The current GlobalPRM object.
        :param local_prms: List of LocalPRM objects.
        :param blocks: List of Block objects.
        :param robot: The Robot object.
        :param goal: Tuple (x, y, z) for the goal location.
        :param goal_threshold: Threshold distance in XY to consider a node as goal.
        :param valid_gaps: List of unique valid gap pairs (each as (i, j)).
        :param planes: List of Plane objects.
        """
        self.global_prm = global_prm
        self.local_prms = local_prms
        self.blocks = blocks
        self.robot = robot
        self.goal = np.array(goal)
        self.goal_threshold = goal_threshold
        self.valid_gaps = valid_gaps
        self.planes = planes
        # Precompute candidate blocks for gaps once.
        if user_candidates is None:
             self.candidate_dict = candidate_blocks_for_gaps(valid_gaps, blocks, planes, robot)
             self._candidates_source = "generated"
        else:
            self.candidate_dict = make_candidate_dict_from_user_names(user_candidates, planes, blocks_list)
            self._candidates_source = "user"
        
        self._bias_idx = 0
        env = build_bfs_snapshot(
            planes=planes,
            global_prm=global_prm,
            blocks=blocks,
            robot_xyz= robot.position,        # current robot position
            goal_xyz=goal,          # current goal position
            candidates=self.candidate_dict, # your superset (gap -> [[block, place_plane], ...])
            permanent_edges=permanent_edges
        )
        
        bfs_start_time = time.perf_counter()
        
        if USE_LLM_BIAS:
            triplets = llm_solve_to_triplets(env)      # uses call_llm stub or cached response
        else:
            triplets = bfs_solve_to_triplets(env)
        
        bfs_end_time = time.perf_counter()
        print(f"BFS found in {bfs_end_time - bfs_start_time:.4f} seconds")
        # triplets = bfs_solve_to_triplets(env)
        print(triplets)
        
        
        
        bias_seq_idx = triplets_to_bias_sequence(triplets, planes=self.planes, blocks=self.blocks)
        
        self.bias_sequence = [ (gap_pair, b_i, self.planes[p_i]) for (gap_pair, b_i, p_i) in bias_seq_idx ]
        
        self.plan_bank = [list(self.bias_sequence)]  # store every BFS plan we ever generate
        self.current_plan_id = 0                     # the index into plan_bank for the active plan



        
        self.tree = []  # List of TreeNode objects

    def find_goal_node(self):
        """
        Vectorized: Return a node index from global_prm that is within goal_threshold (XY) and
        has z equal (within tolerance) to goal's z.
        """
        tol = 1e-3
        same_z = np.abs(self.global_prm.nodes[:, 2] - self.goal[2]) < tol
        if not np.any(same_z):
            return None
        candidate_indices = np.where(same_z)[0]
        candidates_xy = self.global_prm.nodes[same_z, :2]
        distances = np.linalg.norm(candidates_xy - self.goal[:2], axis=1)
        valid = np.where(distances <= self.goal_threshold)[0]
        if valid.size > 0:
            return int(candidate_indices[valid[0]])
        return None
    def _tag_child_node(self, node, used_plan: bool, parent_node=None):
        """
        used_plan=True  -> node is affiliated with the parent's BFS plan (advance cursor by +1).
        used_plan=False -> node is off-plan (random or other), no plan affiliation.
        """
        if used_plan and parent_node is not None and getattr(parent_node, "plan_id", None) is not None:
            setattr(node, "plan_id", parent_node.plan_id)
            # parent_node.plan_cursor is how far parent progressed; child is next step
            setattr(node, "plan_cursor", getattr(parent_node, "plan_cursor", 0) + 1)
        else:
            setattr(node, "plan_id", None)
            setattr(node, "plan_cursor", -1)
    def _all_candidate_union(self):
        """
        Returns a list of tuples: [(gap_pair_or_None, block_idx, place_plane_obj), ...]
        Built from self.candidates which is {(i,j): [(block_idx, place_plane_obj), ...]}.
        """
        U = []
        for gap_pair, lst in self.candidate_dict.items():
            for (b, plane_obj) in lst:
                U.append((gap_pair, b, plane_obj))
        return U

    
    def _random_plane_node(self, prm, plane_name, blocks, require_connected=True, max_tries=64):
        """
        Return a random node index from prm that belongs to plane_name and (optionally)
        is 'live' (has degree > 0) and not under any block on that plane.
        """
        # candidates on the plane
        cand = [
            i for i, m in enumerate(prm.metadata)
            if m.get("source_type") == "Plane" and m.get("source") == plane_name
        ]
        if not cand:
            return None
    
        if not require_connected:
            return random.choice(cand)
    
        # filter: degree > 0 and not under a block at this node's z
        live = []
        for i in cand:
            if prm.nx_graph.degree(i) <= 0:
                continue
            pos = prm.nodes[i]
            if is_under_block(pos, blocks, pos[2]):
                continue
            live.append(i)
    
        if not live:
            return None
    
        # sample with a few retries in case graph mutates between checks
        for _ in range(max_tries):
            i = random.choice(live)
            if prm.nx_graph.has_node(i) and prm.nx_graph.degree(i) > 0:
                return i
        return None

    def _exclude_triplet_and_request_replan(self, parent_node, gap, b_idx, p_idx):
        """
        Add (gap, b_idx, p_idx) to exclusions and request a one-shot replan.
        Seeds the replan from ROOT (forced pick once), as discussed.
        """
        
        if FREEZE_BFS_PLAN:
            # In the experiment: do NOT exclude and do NOT request replan.
            # You can still increment counters for logging if you want.
            return
        
        if not hasattr(self, "replan_exclusions"):
            self.replan_exclusions = set()
        self.replan_exclusions.add((gap, b_idx, p_idx))
    
        # Force next parent selection to be the root, then replan there.
        if self.tree:
            self._force_pick_node_once = self.tree[0]  # swap to self.root_node if you store it explicitly
        self._pending_replan = True  # handled at top of next iteration
    
    
    def _register_triplet_failure(self, parent_node, gap, b_idx, p_idx, kind):
        """
        Increment failure counters per (parent-node, triplet) and failure kind ∈ {'pick','drop','place'}.
        If any counter hits TRIPLET_FAIL_THRESHOLD, exclude triplet and request replan.
        If the triplet is already excluded, do nothing (avoid redundant replans).
        """
        if kind not in ("pick", "drop", "place"):
            return
    
        # If already excluded, don't keep counting/replanning
        if hasattr(self, "replan_exclusions") and (gap, b_idx, p_idx) in self.replan_exclusions:
            return
    
        # Lazily create the per-run registry
        if not hasattr(self, "triplet_fail_counts"):
            self.triplet_fail_counts = {}  # { node_id: { (gap,b,p): {'pick':int,'drop':int,'place':int} } }
    
        node_key = id(parent_node)
        tkey = (gap, b_idx, p_idx)
    
        node_map = self.triplet_fail_counts.setdefault(node_key, {})
        counts = node_map.setdefault(tkey, {"pick": 0, "drop": 0, "place": 0})
    
        counts[kind] += 1
        print(f"______ (node {node_key}) triplet {tkey} {kind} cnt = {counts[kind]}")
    
        if counts[kind] >= TRIPLET_FAIL_THRESHOLD:
            self._exclude_triplet_and_request_replan(parent_node, gap, b_idx, p_idx)


    def get_robot_node_index(self):
        """
        Return the index of the global PRM node closest to the robot.
        """
        robot_pos = np.array(self.robot.position)
        distances = np.linalg.norm(self.global_prm.nodes - robot_pos, axis=1)
        min_idx = int(np.argmin(distances))
        return min_idx
    
    def _triplet_matches_expected(self, parent_node, gap, b_idx, p_idx):
        """
        True iff (gap, b_idx, p_idx) equals the parent's next expected plan step.
        Uses the same comparison you use later for `used_plan`.
        """
        if parent_node is None or getattr(parent_node, "plan_id", None) is None:
            return False
        pid = parent_node.plan_id
        pc  = getattr(parent_node, "plan_cursor", 0)
        plan = self.plan_bank[pid] if 0 <= pid < len(self.plan_bank) else []
        if not (0 <= pc < len(plan)):
            return False
        try:
            exp_gap, exp_bidx, exp_pobj = plan[pc]
        except Exception:
            return False
        try:
            exp_pidx = self.planes.index(exp_pobj)
        except ValueError:
            exp_pidx = None
        return (gap == exp_gap) and (b_idx == exp_bidx) and (p_idx == exp_pidx)

    def _register_onplan_bridge_fail(self, parent_node, attempted_triplet):
        """
        Called only when an on-plan attempt fails at bridge-intent verification.
        Increments per-parent-node fail counter, and if threshold reached:
          - sets self._force_random_node_once = True (escape bias once)
          - sets self._pending_replan = True      (run replan on the next chosen seed)
          - adds the triplet to self.replan_exclusions (accumulating set)
        """
        if FREEZE_BFS_PLAN:
            # Just count, don’t exclude or replan in this experiment.
            return
        
        # per-parent-node counter (lazy)
        cnt = getattr(parent_node, "onplan_fail_count", 0) + 1
        parent_node.onplan_fail_count = cnt
        # print("______ cnt    = ",cnt)
        if cnt >= MAX_ONPLAN_FAILS_BEFORE_REPLAN:
            # Force next parent selection to be the ROOT node (one-shot), then replan there.
            if self.tree:
                self._force_pick_node_once = self.tree[0]   # if you store the root elsewhere, swap to self.root / self.root_node
            else:
                self._force_pick_node_once = None
        
            self._pending_replan = True  # run replan_from_node() right after picking that seed
        
            if not hasattr(self, "replan_exclusions"):
                self.replan_exclusions = set()
            self.replan_exclusions.add(attempted_triplet)


    
    def replan_from_node(self, seed_node):
        """
        Build a BFS snapshot from seed_node, excluding any triplets in self.replan_exclusions,
        run BFS→triplets, convert to bias sequence (with plane OBJECTs), append to plan_bank,
        set current_plan_id, and tag seed_node as the root (cursor=0).
        """
        global BFS_counter
        
        # Accumulating exclusions (triplets in index space): (gap_pair, b_idx, p_idx)
        exclusions = getattr(self, "replan_exclusions", set())
        
    
        # Filter a COPY of candidate_dict for this replan (preserve global superset)
        filtered_candidates = {}
        for gap_key, pairs in self.candidate_dict.items():
            kept = []
            for (b_idx, place_plane_obj) in pairs:
                try:
                    p_idx = self.planes.index(place_plane_obj)
                except ValueError:
                    # If plane object not found in self.planes, skip this pair
                    continue
                if (gap_key, b_idx, p_idx) in exclusions:
                    continue
                kept.append((b_idx, place_plane_obj))
            filtered_candidates[gap_key] = kept
    
        # Seed world state from the node
        seed_prm = seed_node.global_prm
        robot_xyz = tuple(seed_prm.nodes[seed_node.robot_node_index])
        goal_xyz  = tuple(self.goal)
    
        # Permanent plane connectivity
        permanent_edges = compute_permanent_connectivity(seed_prm, self.planes)
    
        # Build BFS snapshot (uses adapter's canonicalization internally)
        env = build_bfs_snapshot(
            planes=self.planes,
            global_prm=seed_prm,
            blocks=seed_node.block_config,
            robot_xyz=robot_xyz,
            goal_xyz=goal_xyz,
            candidates=filtered_candidates,
            permanent_edges=permanent_edges
        )
    
        # Solve BFS → triplets → bias sequence (indices)
        triplets = bfs_solve_to_triplets(env)
        BFS_counter=BFS_counter+1
        print(triplets)
        if not triplets:
            return None
    
        bias_seq_idx = triplets_to_bias_sequence(triplets, planes=self.planes, blocks=self.blocks)
        # Convert plane indices to plane OBJECTs, as your plan entries expect
        new_bias_sequence = [(gap_pair, b_i, self.planes[p_i]) for (gap_pair, b_i, p_i) in bias_seq_idx]
    
        # Append to plan_bank (list of plans), set current plan id
        self.plan_bank.append(list(new_bias_sequence))
        self.current_plan_id = len(self.plan_bank) - 1
    
        # Tag the seed node as root of this new plan
        seed_node.plan_id = self.current_plan_id
        seed_node.plan_cursor = 0
    
        return self.current_plan_id



    def sample_candidate_from_candidates(self, candidate_nodes,bottom_z=None):
        """
        candidate_nodes: list of candidate node indices in global_prm.
        If candidates come from more than one distinct z value, first randomly choose one z, 
        then randomly pick one candidate among those with that z.
        """
        if not candidate_nodes:
            return None
        # Get the z coordinates for each candidate.
        z_vals = [self.global_prm.nodes[i][2] for i in candidate_nodes]
        unique_z = list(set(z_vals))
        if bottom_z:
            chosen_z = bottom_z
        else:
            chosen_z = random.choice(unique_z)
        
        candidates_with_z = [i for i, z in zip(candidate_nodes, z_vals) if abs(z - chosen_z)<1e-3]
        if candidates_with_z:
            return random.choice(candidates_with_z)
        return None

    # def sample_pick_node(self, block):
    #     """
    #     Sample a pick node that is within robot.reach_range of block's bottom centroid.
    #     Return the candidate node index.
    #     """
    #     # block_center = np.array([block.centroid[0], block.centroid[1], block.centroid[2] + block.height/2])
    #     block_center = np.array([block.centroid[0], block.centroid[1], block.centroid[2]])
    #     candidate_nodes = [i for i, node in enumerate(self.global_prm.nodes)
    #                    if np.linalg.norm(node[:3] - block_center[:3]) <= self.robot.reach_range]
    #     # Remove candidate if it is (almost) equal to the block's top centroid.
    #     candidate_nodes = [i for i in candidate_nodes
    #                if not np.allclose(self.global_prm.nodes[i], np.array(block.topcentroid), atol=1e-3)]
    
    #     return self.sample_candidate_from_candidates(candidate_nodes)
    def sample_pick_node(self, block, tol=1e-6):
        """
        Sample a pick node within robot.reach_range of the block's *bottom* centroid,
        excluding: (a) nodes under the block footprint at the bottom z,
                   (b) the block-top node itself,
                   (c) any block nodes (we only want plane nodes to stand on).
        """
        # """
        # SPECIAL EXPERIMENT:
        # Always use the PRM node located at (0,0,0) as the pick node.
        # Returns: integer node index, or None if that node doesn't exist.
        # """
        # nodes = self.global_prm.nodes
        # target = np.array([0.0, 0.0, 0.0])
    
        # # Find all nodes whose position is (0,0,0) up to a small tolerance.
        # mask = np.all(np.isclose(nodes, target, atol=1e-6), axis=1)
        # idxs = np.where(mask)[0]
    
        # if len(idxs) == 0:
        #     print("WARNING: no PRM node at (0,0,0); cannot use it as pick node.")
        #     return None  # planner.plan() already treats None as "resample"
        
        # pick_idx = int(idxs[0])
        # return pick_idx
        
        nodes = self.global_prm.nodes
        metas = self.global_prm.metadata
    
        block_bottom = np.array([block.centroid[0], block.centroid[1], block.centroid[2]])
        block_top    = np.array(block.topcentroid)
        bottom_z     = block.centroid[2]
    
        # 0) plane-only mask (never stand on block nodes for pick)
        plane_only_mask = np.array([m.get("source_type") == "Plane" for m in metas])
    
        # 1) 3D reach to bottom centroid
        d3 = np.linalg.norm(nodes - block_bottom, axis=1)
        reach_mask = d3 <= self.robot.reach_range
    
        # 2) row-wise equality to block-top node
        equal_top_mask = np.all(np.isclose(nodes, block_top, atol=1e-3), axis=1)
        not_top_mask = ~equal_top_mask
    
        # 3) exclude nodes under the block footprint on the bottom plane
        same_z_mask = np.abs(nodes[:, 2] - bottom_z) < tol
        bbox_mask   = _points_in_bbox_mask(nodes[:, :2], block.bounding_box.bounds)
        under_fast  = same_z_mask & bbox_mask
        if np.any(under_fast):
            idxs = np.where(under_fast)[0]
            precise = np.array([block.bounding_box.contains(Point(nodes[i,0], nodes[i,1])) for i in idxs])
            under_mask = np.zeros(nodes.shape[0], dtype=bool)
            under_mask[idxs] = precise
        else:
            under_mask = under_fast  # all False
    
        cand_mask = plane_only_mask & reach_mask & not_top_mask & (~under_mask)
        candidate_nodes = np.where(cand_mask)[0].tolist()
    
        return self.sample_candidate_from_candidates(candidate_nodes)

    # def sample_drop_node(self, block):
    #     """
    #     Sample a drop node that is within robot.reach_range of block's top centroid.
    #     Return the candidate node index.
    #     """
    #     block_center = np.array([block.centroid[0], block.centroid[1], block.centroid[2] + block.height/2])
    #     candidate_nodes = [i for i, node in enumerate(self.global_prm.nodes)
    #                        if np.linalg.norm(node[:2] - block_center[:2]) <= self.robot.reach_range]
    #     # Remove candidate if it is (almost) equal to the block's top centroid.
    #     candidate_nodes = [i for i in candidate_nodes
    #                if not np.allclose(self.global_prm.nodes[i], np.array(block.topcentroid), atol=1e-3)]
    #     return self.sample_candidate_from_candidates(candidate_nodes)
    def sample_drop_node(self, block, tol=1e-6):
        """
        Sample a drop node within robot.reach_range of the block's *top* centroid,
        excluding: (a) nodes under the block footprint at the bottom z,
                   (b) the block-top node itself,
                   (c) any block nodes (we only want plane nodes to stand on while placing).
        """
        
        # """
        # SPECIAL EXPERIMENT:
        # Always use the PRM node located at (0,0,0) as the pick node.
        # Returns: integer node index, or None if that node doesn't exist.
        # """
        # nodes = self.global_prm.nodes
        # target = np.array([0.0, 0.0, 0.0])
    
        # # Find all nodes whose position is (0,0,0) up to a small tolerance.
        # mask = np.all(np.isclose(nodes, target, atol=1e-6), axis=1)
        # idxs = np.where(mask)[0]
    
        # if len(idxs) == 0:
        #     print("WARNING: no PRM node at (0,0,0); cannot use it as pick node.")
        #     return None  # planner.plan() already treats None as "resample"
        
        # pick_idx = int(idxs[0])
        # return pick_idx
        
        nodes = self.global_prm.nodes
        metas = self.global_prm.metadata
    
        block_top   = np.array(block.topcentroid)
        bottom_z    = block.centroid[2]
    
        # 0) plane-only mask
        plane_only_mask = np.array([m.get("source_type") == "Plane" for m in metas])
    
        # 1) XY reach to top centroid
        dxy = np.linalg.norm(nodes[:, :2] - block_top[:2], axis=1)
        reach_mask = dxy <= self.robot.reach_range
    
        # 2) row-wise equality to block-top node
        equal_top_mask = np.all(np.isclose(nodes, block_top, atol=1e-3), axis=1)
        not_top_mask = ~equal_top_mask
    
        # 3) exclude plane nodes under the block footprint (at bottom z)
        same_z_mask = np.abs(nodes[:, 2] - bottom_z) < tol
        bbox_mask   = _points_in_bbox_mask(nodes[:, :2], block.bounding_box.bounds)
        under_fast  = same_z_mask & bbox_mask
        if np.any(under_fast):
            idxs = np.where(under_fast)[0]
            precise = np.array([block.bounding_box.contains(Point(nodes[i,0], nodes[i,1])) for i in idxs])
            under_mask = np.zeros(nodes.shape[0], dtype=bool)
            under_mask[idxs] = precise
        else:
            under_mask = under_fast
    
        cand_mask = plane_only_mask & reach_mask & not_top_mask & (~under_mask)
        candidate_nodes = np.where(cand_mask)[0].tolist()
    
        return self.sample_candidate_from_candidates(candidate_nodes,bottom_z)
    
    def _choose_tree_node(self):
        """
        Node selector f_V using global p_node_bias:
          - If self._force_pick_node_once is set: return that node once, then clear the flag.
          - Else if self._force_random_node_once is set: pick uniformly once, then clear the flag.
          - Else with prob p_node_bias: prefer the most-advanced node in current_plan_id,
            falling back to most-advanced across all plans.
          - Otherwise: uniform over the entire tree.
        """
        if not self.tree:
            return None
    
        # Highest-priority one-shot: pick a specific node (e.g., root) if requested
        forced = getattr(self, "_force_pick_node_once", None)
        if forced is not None:
            self._force_pick_node_once = None
            return forced
    
        # One-iteration escape hatch: bypass biasing once, then clear the flag.
        if getattr(self, "_force_random_node_once", False):
            self._force_random_node_once = False
            return random.choice(self.tree)
    
        use_bias = (random.random() < p_node_bias)
    
        if use_bias:
            # Prefer the currently active plan's most-advanced nodes first.
            cur_pid = getattr(self, "current_plan_id", None)
            if cur_pid is not None:
                cur_plan_nodes = [(getattr(n, "plan_cursor", -1), n)
                                  for n in self.tree if getattr(n, "plan_id", None) == cur_pid]
                if cur_plan_nodes:
                    max_prog = max(pc for pc, _ in cur_plan_nodes)
                    best = [n for pc, n in cur_plan_nodes if pc == max_prog]
                    return random.choice(best)
    
            # Fallback: most-advanced among all planned nodes.
            planned = [(getattr(n, "plan_cursor", -1), n)
                       for n in self.tree if getattr(n, "plan_id", None) is not None]
            if planned:
                max_prog = max(pc for pc, _ in planned)
                best = [n for pc, n in planned if pc == max_prog]
                return random.choice(best)
    
        # Fallback: uniform over all nodes
        return random.choice(self.tree)




    def select_biased_triplet(self, parent_node):
        """
        Returns:
          (gap_pair_or_None, block_idx, place_plane_idx)
    
        Buckets:
          - plan-guided (next step for parent's plan, if any)
          - gap-guided (uniform from candidates of that next gap)
          - random      (uniform from the union across all gaps)
    
        Notes:
          - Does NOT advance any cursor. Advancement happens only after successful expansion.
          - Falls through if a chosen bucket is unavailable/empty.
          - Raises ValueError only if there are literally zero candidates overall.
        """
        # ----- Determine the plan-guided "next step" for this parent (if affiliated) -----
        next_step = None
        if parent_node is not None and getattr(parent_node, "plan_id", None) is not None:
            pid = parent_node.plan_id
            pc  = getattr(parent_node, "plan_cursor", 0)
            plan = self.plan_bank[pid] if 0 <= pid < len(self.plan_bank) else []
            if 0 <= pc < len(plan):
                # next step is plan[pc]
                # expected shape: (gap_pair_or_None, block_idx, place_plane_obj)
                next_step = plan[pc]
    
        # ----- Build the random-union once (used in multiple fallbacks) -----
        union_all = self._all_candidate_union()
    
        if not union_all:
            # No candidates at all -> upstream will (later) trigger a BFS replan
            raise ValueError("No candidates available (union empty).")
    
        # ----- Normalize bucket probabilities (ensure >0 and sum to 1) -----
        eps = 1e-12
        a = max(p_triplet_plan, eps)
        b = max(p_triplet_gap,  eps)
        c = max(p_triplet_rand, eps)
        s = a + b + c
        a, b, c = a/s, b/s, c/s
    
        # ----- Define local bucket attempts in order based on sampled bucket, with fall-through -----
        def try_plan_guided():
            if next_step is None:
                return None
            (gap_or_None, block_idx, place_plane_obj) = next_step
            # Map plane object to its index
            try:
                plane_idx = self.planes.index(place_plane_obj)
            except ValueError:
                return None
            return (gap_or_None, block_idx, plane_idx)
    
        def try_gap_guided():
            if next_step is None:
                return None
            gap = next_step[0]
            if not (isinstance(gap, tuple) and len(gap) == 2):
                return None
            # Uniform from candidates for this gap
            cand_list = self.candidate_dict.get(gap, [])
            if not cand_list:
                return None
            (b_idx, plane_obj) = random.choice(cand_list)
            try:
                p_idx = self.planes.index(plane_obj)
            except ValueError:
                return None
            return (gap, b_idx, p_idx)
    
        def try_random():
            # Uniform over union
            (gap_any, b_idx, plane_obj) = random.choice(union_all)
            try:
                p_idx = self.planes.index(plane_obj)
            except ValueError:
                # Extremely unlikely given consistency; try a few more times
                for _ in range(8):
                    (gap_any, b_idx, plane_obj) = random.choice(union_all)
                    try:
                        p_idx = self.planes.index(plane_obj)
                        break
                    except ValueError:
                        p_idx = None
                if p_idx is None:
                    return None
            return (gap_any, b_idx, p_idx)
    
        # Sample a bucket, then fall through if needed
        u = random.random()
        if u < a:
            order = (try_plan_guided, try_gap_guided, try_random)
        elif u < a + b:
            order = (try_gap_guided, try_plan_guided, try_random)
        else:
            order = (try_random, try_plan_guided, try_gap_guided)
    
        for fn in order:
            res = fn()
            if res is not None:
                return res
    
        # Should not reach here, but as a last safeguard try random again
        res = try_random()
        if res is not None:
            return res
    
        # If literally nothing was returned despite nonempty union, raise
        raise ValueError("Triplet chooser failed to select despite available candidates.")



    def plan(self, max_expansions=5000, return_reachability=False):
        """
        Build a plan to reach the goal.
        First, check for a trivial path in the current global PRM.
        If not found, expand the tree until a path from the drop node to the goal is found,
        or max_expansions is reached.
        
        Returns a sequence of high-level actions in the form:
          [("GoTo", [(x1,y1), (x2,y2), ...]),
           ("Pick", block_index),
           ("GoTo", [(x3,y3), ...]),
           ("Drop", (x_drop, y_drop)),
           ("GoTo", [(x_goal,y_goal)])]
        """
        # First, attempt a trivial connectivity check.
        start_robot_node = self.get_robot_node_index()
        goal_node = self.find_goal_node()
        if goal_node is not None:
            try:
                trivial_path = nx.astar_path(self.global_prm.nx_graph, start_robot_node, goal_node,
                                             heuristic=lambda a, b: np.linalg.norm(self.global_prm.nodes[a]-self.global_prm.nodes[b]))
                trivial_coords = [tuple(self.global_prm.nodes[i]) for i in trivial_path]
                if return_reachability:
                    reach = [get_reachable_planes_from_robot(self.global_prm, goal_node) if goal_node is not None else set()]
                    return [("GoTo", trivial_coords)], reach
                return [("GoTo", trivial_coords)]
            except nx.NetworkXNoPath:
                pass  # Proceed with tree expansion.
        
        # Initialize the tree with the initial state.
        initial_prm = copy.deepcopy(self.global_prm)
        root = TreeNode(start_robot_node, initial_prm, copy.deepcopy(self.blocks), [], reach_hist=[])
        self.tree = [root]
        # Tag root with the current plan
        setattr(root, "plan_id", self.current_plan_id)
        setattr(root, "plan_cursor", 0)   # progress along the bias_sequence for this node/branch
        
        expansions = 0
        new_Node=None
        
    
        
        while self.tree and expansions < max_expansions:
            start_time = time.perf_counter()
            print("iter:",expansions) 
            print("iter:", expansions)
            current_Node = self._choose_tree_node()
            
            # If a replan was requested last iter, run it now on this one-shot random seed.
            if getattr(self, "_pending_replan", False) and not FREEZE_BFS_PLAN:
                self.replan_from_node(current_Node)
                self._pending_replan = False

            current_robot_node = current_Node.robot_node_index
            current_prm = current_Node.global_prm
            current_blocks = copy.deepcopy(current_Node.block_config)
            current_actions = current_Node.actions
            
            new_prm = current_prm.clone()
            for i, meta in enumerate(new_prm.metadata):
                if meta.get("source_type") == "Plane":
                    meta["blocks"] = current_blocks  # the per-branch Block objects
            
            
            # Choose a triplet based on the parent node’s plan affiliation (3-bucket bias)
            gap, candidate_idx, candidate_plane_idx = self.select_biased_triplet(current_Node)
            candidate_plane = self.planes[candidate_plane_idx]
            candidate_block = current_blocks[candidate_idx]
            print("triplet:",gap, candidate_idx, candidate_plane_idx)
            # Tag this attempt as "on-plan" or not (used only for failure accounting/replan triggers)
            is_onplan_attempt = self._triplet_matches_expected(current_Node, gap, candidate_idx, candidate_plane_idx)
            # --- Universal exclusion guard: never attempt blacklisted triplets, regardless of bucket ---
            if hasattr(self, "replan_exclusions") and (gap, candidate_idx, candidate_plane_idx) in self.replan_exclusions:
                # Try to resample within the SAME gap first (if any), excluding all blacklisted triplets
                def _pick_alt_for_gap(exp_gap):
                    opts = []
                    for (bi, pobj) in self.candidate_dict.get(exp_gap, []):
                        try:
                            pi = self.planes.index(pobj)
                        except ValueError:
                            continue
                        if (exp_gap, bi, pi) in self.replan_exclusions:
                            continue
                        opts.append((exp_gap, bi, pi))
                    return random.choice(opts) if opts else None
            
                choice = None
                if gap is not None:
                    choice = _pick_alt_for_gap(gap)
            
                if choice is None:
                    # Fall back to union across ALL candidates, excluding blacklisted triplets
                    union_opts = []
                    for gk, pairs in self.candidate_dict.items():
                        for (bi, pobj) in pairs:
                            try:
                                pi = self.planes.index(pobj)
                            except ValueError:
                                continue
                            if (gk, bi, pi) in self.replan_exclusions:
                                continue
                            union_opts.append((gk, bi, pi))
                    if union_opts:
                        choice = random.choice(union_opts)
            
                if choice is None:
                    # Nothing valid to try this iteration
                    expansions += 1
                    continue
            
                # Adopt the alternate choice
                gap, candidate_idx, candidate_plane_idx = choice
                candidate_plane = self.planes[candidate_plane_idx]
                candidate_block = current_blocks[candidate_idx]
                is_onplan_attempt = False  # no longer the plan step; don't attribute failures to the plan
            # --- end universal exclusion guard ---

            
            # --- Pick Phase ---
            pick_attempt = 0
            pick_node = None
            while pick_attempt < 30 and pick_node is None:
                pick_node_candidate = self.sample_pick_node(candidate_block)
                if pick_node_candidate is None:
                    pick_attempt += 1
                    continue
                try:
                    path_to_pick = nx.astar_path(new_prm.nx_graph, current_robot_node, pick_node_candidate,
                                                 heuristic=lambda a, b: np.linalg.norm(new_prm.nodes[a]-new_prm.nodes[b]))
                    pick_node = pick_node_candidate
                except nx.NetworkXNoPath:
                    pick_attempt += 1
            # if pick_node is None:
            #     expansions += 1
            #     continue  # Restart expansion.
            if pick_node is None:
                self._register_triplet_failure(current_Node, gap, candidate_idx, candidate_plane_idx, kind="pick")
                expansions += 1
                continue

            new_prm.fast_on_pick(candidate_block)
            prm_after_pick = new_prm.clone()  # snapshot after pick, before place
            
            # --- Drop Phase ---
            drop_attempt = 0
            drop_node = None
            # drop_point = sample_block_placement_general(gap, candidate_plane, candidate_block, self.robot, self.planes, current_blocks)
            drop_point = sample_block_placement_general(gap,candidate_plane,candidate_block,self.robot,
                self.planes,current_blocks,current_prm=current_prm,max_attempts=25)
            
            if drop_point is None:
                # Count a "place" failure for this (parent, triplet)
            
                # Only treat as an on-plan *bridge* failure if we were actually trying to bridge a gap.
                if is_onplan_attempt and isinstance(gap, tuple) and len(gap) == 2:
                    self._register_triplet_failure(current_Node, gap, candidate_idx, candidate_plane_idx, kind="place")
                    self._register_onplan_bridge_fail(current_Node, (gap, candidate_idx, candidate_plane_idx))
            
                expansions += 1
                continue

           
            candidate_block.update_position(drop_point)
            while drop_attempt < 30 and drop_node is None:
                drop_node_candidate = self.sample_drop_node(candidate_block)
                if drop_node_candidate is None:
                    drop_attempt += 1
                    continue
                try:
                    path_to_drop = nx.astar_path(new_prm.nx_graph, pick_node, drop_node_candidate,
                                                 heuristic=lambda a, b: np.linalg.norm(new_prm.nodes[a]-new_prm.nodes[b]))
                    drop_node = drop_node_candidate
                except nx.NetworkXNoPath:
                    drop_attempt += 1
            # if drop_node is None:
            #     expansions += 1
            #     continue  # Restart expansion.
            if drop_node is None:
                self._register_triplet_failure(current_Node, gap, candidate_idx, candidate_plane_idx, kind="drop")
                expansions += 1
                continue

            # --- Update Phase ---
            # Simulate moving the block by updating its configuration.
            candidate_block.update_position(drop_point)
            new_prm.fast_on_place(candidate_block, drop_point)
            
            

            # --- Bridge-intent verification (only if a specific gap was intended) ---
            if isinstance(gap, tuple):
                i, j = gap
                p1_name = self.planes[i].name
                p2_name = self.planes[j].name
                # pick representatives that are (a) on-plane, (b) degree>0, (c) not under a block
                n1 = self._random_plane_node(new_prm, p1_name, current_blocks, require_connected=True)
                n2 = self._random_plane_node(new_prm, p2_name, current_blocks, require_connected=True)
                if n1 is None or n2 is None:
                    expansions += 1
                    continue
                try:
                    _ = nx.astar_path(
                        new_prm.nx_graph, n1, n2,
                        heuristic=lambda a, b: np.linalg.norm(new_prm.nodes[a] - new_prm.nodes[b])
                    )
                    # bridged ✅
                except nx.NetworkXNoPath:
                    # If this failure came from the parent's plan-guided next step, register it
                    if is_onplan_attempt:
                        self._register_onplan_bridge_fail(current_Node, (gap, candidate_idx, candidate_plane_idx))
                    expansions += 1
                    continue



           
            # Convert paths to coordinates.
            path_to_pick_coords = [tuple(current_prm.nodes[i]) for i in path_to_pick]
            path_to_drop_coords = [tuple(new_prm.nodes[i]) for i in path_to_drop]
            
            # For "Pick", record the block's index in self.blocks.
            block_index = candidate_idx
            
            new_actions = current_actions + [("GoTo", path_to_pick_coords),
                                             ("Pick", block_index),
                                             ("GoTo", path_to_drop_coords),
                                             ("Drop", (drop_point[0], drop_point[1], drop_point[2]))]
            # --- Reachability timeline bookkeeping (for visualization) ---
            # current_Node.reach_hist is aligned with current_Node.actions.
            # We append reachability after each new elementary action we just constructed.
            reach_hist = list(getattr(current_Node, "reach_hist", []))

            # After GoTo -> pick_node (uses current_prm before picking)
            reach_hist.append(get_reachable_planes_from_robot(current_prm, pick_node))
            # After Pick (uses prm_after_pick where the block has been removed)
            reach_hist.append(get_reachable_planes_from_robot(prm_after_pick, pick_node))
            # After GoTo -> drop_node (still before placing)
            reach_hist.append(get_reachable_planes_from_robot(prm_after_pick, drop_node))
            # After Drop (uses new_prm after placing)
            reach_hist.append(get_reachable_planes_from_robot(new_prm, drop_node))

            new_Node = TreeNode(drop_node, new_prm, current_blocks, new_actions, reach_hist=reach_hist, parent=current_Node)
            self.tree.append(new_Node)
            # Decide whether this success matched the parent's actual plan next step
            used_plan = False
            if current_Node is not None and getattr(current_Node, "plan_id", None) is not None:
                pid = current_Node.plan_id
                pc  = getattr(current_Node, "plan_cursor", 0)
                plan = self.plan_bank[pid] if 0 <= pid < len(self.plan_bank) else []
                if 0 <= pc < len(plan):
                    exp_gap, exp_bidx, exp_pobj = plan[pc]
                    # Compare chosen triplet with the expected next step
                    # We compare by plane object identity or, more robustly, by idx
                    try:
                        exp_pidx = self.planes.index(exp_pobj)
                    except ValueError:
                        exp_pidx = None
                    if (gap == exp_gap) and (candidate_idx == exp_bidx) and (candidate_plane_idx == exp_pidx):
                        used_plan = True
            
            # Tag the child accordingly (advances plan_cursor by +1 only if used_plan=True)
            self._tag_child_node(new_Node, used_plan=used_plan, parent_node=current_Node)

            # visualize_global_prm_reachable(
            #     new_prm,
            #     planes,
            #     current_blocks,
            #     robot_node_index=new_Node.robot_node_index,
            #     highlight_reachable=True,
            #     title="Initial Environment with Global PRM")
            # visualize_global_prm(new_prm, planes, current_blocks, title="Initial Environment with Global PRM")
            
            end_time = time.perf_counter()
            print(f" iteration done in {end_time - start_time:.4f} seconds")
            
            # Now, check if a path exists from the drop node to the goal in new_prm.
            # goal_node = self.find_goal_node()  # Evaluate on the updated global PRM.
            if goal_node is not None:
                try:
                    path_to_goal = nx.astar_path(new_prm.nx_graph, drop_node, goal_node,
                                                 heuristic=lambda a, b: np.linalg.norm(new_prm.nodes[a]-new_prm.nodes[b]))
                    path_to_goal_coords = [tuple(new_prm.nodes[i]) for i in path_to_goal]
                    solution_actions = new_actions + [("GoTo", path_to_goal_coords)]
                    if return_reachability:
                        sol_reach = list(reach_hist) if "reach_hist" in locals() else []
                        sol_reach.append(get_reachable_planes_from_robot(new_prm, goal_node))
                        return solution_actions, sol_reach
                    return solution_actions
                except nx.NetworkXNoPath:
                    pass
            
            expansions += 1
        if return_reachability:
            return None, None
        return None



if __name__ == '__main__':
    start_time = time.perf_counter()
    
    
    
    
    
    
    
    
    """ 6 horizon """
    
    ground_vertices = [(-2, 0, 0), (30, 0, 0), (30, 30, 0), (-2, 30, 0)]
    table1_vertices = [(2, 8, 2), (12, 8, 2), (12, 20, 2), (2, 20, 2)]
    table2_vertices = [(20, 11, 2), (29, 11, 2), (29, 24, 2), (20, 24, 2)]
    O_vertices = [(9, 14, 3), (12, 14, 3), (12, 18, 3), (9, 18, 3)]
    Y_vertices = [(2, 14, 5), (6, 14, 5), (6, 18, 5), (2, 18, 5)]
    R_vertices = [(20, 18, 4), (25, 18, 4), (25, 24, 4), (20, 24, 4)]
    
    
    
    ground_plane = Plane("Ground", ground_vertices)
    table1_plane = Plane("Table1", table1_vertices)
    table2_plane = Plane("Table2", table2_vertices)
    O_plane = Plane("O", O_vertices)
    Y_plane = Plane("Y", Y_vertices)
    R_plane = Plane("R", R_vertices)
    # P_plane = Plane("P", P_vertices)
    planes = [ground_plane, table1_plane, table2_plane, O_plane, Y_plane, R_plane]
    # planes = [ground_plane, table1_plane, table2_plane, O_plane, Y_plane, R_plane, P_plane]
    add_higher_plane_obstacles(planes)
     
    # (Optional) Add higher plane obstacles if needed.
    # add_higher_plane_obstacles([ground_plane, table1_plane, table2_plane])
    
    
    # Create the block and the robot. # Create two blocks.
    block1 = Block(centroid=(18, 3, 0), length=2, width=2, height=1, color="orange", name="B1")
    block2 = Block(centroid=(8, 11, 2), length=2, width=2, height=1, color="red", name="B2")
    block3 = Block(centroid=(21, 23, 4), length=2, width=2, height=2, color="green", name="B3")
    
    blocks_list = [block1,block2,block3]
    
    robot = Robot(position=(0, 0, 0), jump_height=1.2, horizontal_range=3, move_range=3, reach_range=1.9)
    
    # Build local PRMs.
    local_ground = LocalPRM(ground_plane, connection_threshold=2.0, all_blocks=blocks_list)
    local_table1 = LocalPRM(table1_plane, connection_threshold=2.0, all_blocks=blocks_list)
    local_table2 = LocalPRM(table2_plane, connection_threshold=2.0, all_blocks=blocks_list)
    local_O = LocalPRM(O_plane, connection_threshold=2.0, all_blocks=blocks_list)
    local_Y = LocalPRM(Y_plane, connection_threshold=2.0, all_blocks=blocks_list)
    local_R = LocalPRM(R_plane, connection_threshold=2.0, all_blocks=blocks_list)
    # local_P = LocalPRM(P_plane, connection_threshold=2.0, all_blocks=blocks_list)
    
    # Sample nodes on each plane.
    local_ground.sample_nodes(450)
    local_table1.sample_nodes(75)
    local_table2.sample_nodes(50)
    local_O.sample_nodes(5)
    local_Y.sample_nodes(10)
    local_R.sample_nodes(20)
    # local_P.sample_nodes(15)
    
    # Connect nodes in each local PRM.
    local_ground.connect_nodes()
    local_table1.connect_nodes()
    local_table2.connect_nodes()
    local_O.connect_nodes()
    local_Y.connect_nodes()
    local_R.connect_nodes()
    # local_P.connect_nodes()
    
    # Build the global PRM.
    global_prm = GlobalPRM(jump_threshold=1.2, horizontal_range=3, block_extra_factor=1.5)
    global_prm.build_from_local([local_ground, local_table1, local_table2, local_O, local_Y, local_R], blocks_list)
    # global_prm.build_from_local([local_ground, local_table1, local_table2, local_O, local_Y, local_R, local_P], blocks_list)
    
    permanent_edges = compute_permanent_connectivity(global_prm, planes)

    
    end_time = time.perf_counter()
    print(f"Global PRM built in {end_time - start_time:.4f} seconds")
    
    # Visualize the initial environment.
    # visualize_global_prm(global_prm, planes, blocks_list, title="Initial Environment with Global PRM")
    
    
    start_time2 = time.perf_counter()
    # For valid_gaps, assume you use find_valid_gaps(planes, dgap) with an appropriate dgap.
    valid_gaps = find_valid_gaps(planes, dgap=5.0)
    print(valid_gaps)
    # valid_gaps=[(0,1),(0,2),(3,4)]
    # user_candidates = None
    # Define user candidates instead of using candidate function if you to manually define candidates
    user_candidates = {
    'Ground-Table1': [['B1', 'Ground'], ['B2', 'Ground']], 
    'Ground-Table2': [['B1', 'Ground'], ['B2', 'Ground']], 
    'O-Y': [['B3', 'Table1']],
    'Table2-R':  [ ['B1', 'Table2'], ['B2', 'Table2']]}
    goal=(4,17,5)
    planner = Planner(global_prm,  [local_ground, local_table1, local_table2, local_O, local_Y, local_R], blocks_list, robot, goal, goal_threshold=2.0, valid_gaps=valid_gaps, planes=planes, permanent_edges=permanent_edges, user_candidates=user_candidates)
    # planner = Planner(global_prm,  [local_ground, local_table1, local_table2, local_O, local_Y, local_R, local_P], blocks_list, robot, goal=(20,19,4), goal_threshold=2.0, valid_gaps=valid_gaps, planes=planes)
    plan_actions, reach_timeline = planner.plan(return_reachability=True)
    end_time = time.perf_counter()
    print(f"Plan found in {end_time - start_time:.4f} seconds")
    #  print("plan_actions=", plan_actions)
    
    print("BFS_counter=", BFS_counter)
    print("number of nodes", len(planner.tree))
    # viz.visualize_bbnamo(planes, blocks_list, plan_actions,
    #                     limits=((-2,30),(0,30),(0,6)), pause_dt=0.15,
    #                     global_prm=global_prm,
    #                     show_reachable_planes=True,
    #                     )
    viz.visualize_bbnamo(planes, blocks_list, plan_actions, goal,
                         limits=((-2,30),(0,30),(0,6)), pause_dt=0.25,
                         reachable_plane_names_timeline=reach_timeline)
  



    """ 2 horizon """
    # # Define planes using your coordinates.
    
    # ground_vertices = [(-2, 0, 0), (30, 0, 0), (30, 20, 0), (-2, 20, 0)]
    # table1_vertices = [(8, 10, 2), (22, 10, 2), (22, 20, 2), (8, 20, 2)]
    # table2_vertices = [(8, 15, 4), (22, 15, 4), (22, 20, 4), (8, 20, 4)]
    
    
    # ground_plane = Plane("Ground", ground_vertices)
    # table1_plane = Plane("Table1", table1_vertices)
    # table2_plane = Plane("Table2", table2_vertices)
    # planes = [ground_plane, table1_plane, table2_plane]
    # add_higher_plane_obstacles(planes)
    

    # # Create the block and the robot. # Create two blocks.
    # block2 = Block(centroid=(18, 2, 0), length=2, width=2, height=1, color="orange", name="B2")
    # block1 = Block(centroid=(23, 6, 0), length=2, width=2, height=3, color="red", name="B1")
    
    # blocks_list = [block1,block2]
    
    # robot = Robot(position=(0, 0, 0), jump_height=1.2, horizontal_range=2, move_range=2, reach_range=1.9)
    
    # # Build local PRMs.
    # local_ground = LocalPRM(ground_plane, connection_threshold=2.0, all_blocks=blocks_list)
    # local_table1 = LocalPRM(table1_plane, connection_threshold=2.0, all_blocks=blocks_list)
    # local_table2 = LocalPRM(table2_plane, connection_threshold=2.0, all_blocks=blocks_list)
    # # local_P = LocalPRM(P_plane, connection_threshold=2.0, all_blocks=blocks_list)
    
    # # Sample nodes on each plane.
    # local_ground.sample_nodes(300)
    # local_table1.sample_nodes(40)
    # local_table2.sample_nodes(20)
    # # local_P.sample_nodes(15)
    
    # # Connect nodes in each local PRM.
    # local_ground.connect_nodes()
    # local_table1.connect_nodes()
    # local_table2.connect_nodes()
    
    # # Build the global PRM.
    # global_prm = GlobalPRM(jump_threshold=1.2, horizontal_range=3, block_extra_factor=1.5)
    # global_prm.build_from_local([local_ground, local_table1, local_table2], blocks_list)
    
    # permanent_edges = compute_permanent_connectivity(global_prm, planes)
    

    # # Define user candidates instead of using candidate function if you to manually define candidates
    # # Specifically used for adding incorrect candidates
    
    # user_candidates = {
    # # 'Ground-Table1': [['B2', 'Ground']],
    # # 'Table1-Table2':  [['B1', 'Table2'],['B2', 'Table2'],['B2', 'Ground']], 
    # # 'Ground-Table2': [],}
    # # user_candidates = {
    # 'Ground-Table1': [['B2', 'Ground'],['B1', 'Ground'] ],
    # 'Table1-Table2':  [['B1', 'Table2'],['B2', 'Table2'],['B1', 'Ground']], 
    # 'Ground-Table2': [['B1', 'Ground'],['B2', 'Ground']],}
    # # user_candidates = {
    # # 'Ground-Table1': [['B1', 'Ground']],
    # # 'Table1-Table2':  [['B1', 'Table1'],['B2', 'Ground']], 
    # # 'Ground-Table2': [],}
    # # user_candidates=None
    
    # end_time = time.perf_counter()
    # print(f"Global PRM built in {end_time - start_time:.4f} seconds")
    
    # # Visualize the initial environment.
    # # visualize_global_prm(global_prm, planes, blocks_list, title="Initial Environment with Global PRM")
    
    # start_time2 = time.perf_counter()
    
    # valid_gaps=[(0,1),(1,2)]
    # goal=(20,19,4)
    # planner = Planner(global_prm,  [local_ground, local_table1, local_table2], blocks_list, robot, goal, goal_threshold=2.0, valid_gaps=valid_gaps, planes=planes, permanent_edges=permanent_edges, user_candidates=user_candidates)
    # # planner = Planner(global_prm,  [local_ground, local_table1, local_table2, local_O, local_Y, local_R, local_P], blocks_list, robot, goal=(20,19,4), goal_threshold=2.0, valid_gaps=valid_gaps, planes=planes, permanent_edges=permanent_edges)
    # plan_actions, reach_timeline = planner.plan(return_reachability=True)
    # end_time = time.perf_counter()
    # print(f"Plan found in {end_time - start_time:.4f} seconds")
    # #  print("plan_actions=", plan_actions)
    # print("number of nodes=", len(planner.tree))
    
    
    # # ... after you have: planes, blocks_list, plan_actions
    # viz.visualize_bbnamo(planes, blocks_list, plan_actions, goal,
    #                      limits=((-2,30),(0,30),(0,6)), pause_dt=0.25,
    #                      reachable_plane_names_timeline=reach_timeline)


    """============================
    Paper figure: 20x20 two-plane scene Figures 1 and 2
    ============================"""
    
    # # --- Planes ---
    # ground_vertices = [(0.0, 0.0, 0.0), (20.0, 0.0, 0.0), (20.0, 20.0, 0.0), (0.0, 20.0, 0.0)]
    # # Plane 1: from (2.5, 7.5) to (17.5, 12.5) @ z=2.0
    # plane1_vertices = [(2.5, 7.5, 1.0), (17.5, 7.5, 1.0), (17.5, 12.5, 1.0), (2.5, 12.5, 1.0)]
    # # Plane 2: from (5.0, 12.5) to (15.0, 17.5) @ z=3.5
    # plane2_vertices = [(5.0, 12.5, 3), (15.0, 12.5, 3), (15.0, 17.5, 3), (5.0, 17.5, 3)]
    # plane3_vertices = [(1.0, 12.5, 5), (5.0, 12.5, 5), (5.0, 17.5, 5), (1.0, 17.5, 5)]
    
    # # ground_plane = Plane("Ground", ground_vertices)
    # # plane1 = Plane("P1", plane1_vertices)
    # # plane2 = Plane("P2", plane2_vertices)
    # # planes = [ground_plane, plane1, plane2]
    
    # ground_plane = Plane("Ground", ground_vertices)
    # plane1 = Plane("P1", plane1_vertices)
    # plane2 = Plane("P2", plane2_vertices)
    # plane3 = Plane("P3", plane3_vertices)
    # planes = [ground_plane, plane1, plane2, plane3]
    
    # # imprint higher planes as obstacles onto lower ones
    # add_higher_plane_obstacles(planes)
    
    # # --- Blocks (one block, 2×2×1.2) ---
    # # block1 = Block(centroid=(2.0, 2.0, 0.0), length=2.0, width=2.0, height=1.2, color="orange", name="B1")
    # # blocks_list = [block1]
    
    # block1 = Block(centroid=(2.0, 2.0, 0.0), length=2.0, width=2.0, height=1, color="orange", name="B1")
    # block2 = Block(centroid=(6.0, 1.0, 0.0), length=2.0, width=2.0, height=1, color="orange", name="B2")
    # blocks_list = [block1,block2]
    
    # # --- Robot (jump height 1.4) ---
    # robot = Robot(position=(18.0, 2.0, 0.0), jump_height=1.4, horizontal_range=3.0, move_range=2.5, reach_range=2.0)
    
    # # --- Local PRMs (densities as needed) ---
    # local_ground = LocalPRM(ground_plane, connection_threshold=2.0, all_blocks=blocks_list)
    # local_p1     = LocalPRM(plane1,       connection_threshold=2.0, all_blocks=blocks_list)
    # local_p2     = LocalPRM(plane2,       connection_threshold=2.0, all_blocks=blocks_list)
    # local_p3     = LocalPRM(plane3,       connection_threshold=2.0, all_blocks=blocks_list)
    
    # local_ground.sample_nodes(300); local_ground.connect_nodes()
    # local_p1.sample_nodes(50);      local_p1.connect_nodes()
    # local_p2.sample_nodes(50);      local_p2.connect_nodes()
    # local_p3.sample_nodes(50);      local_p3.connect_nodes()
    
    # # --- Global PRM (jump threshold 1.4) ---
    # global_prm = GlobalPRM(jump_threshold=1.4, horizontal_range=3.0, block_extra_factor=1.5)
    # global_prm.build_from_local([local_ground, local_p1, local_p2, local_p3], blocks_list)
    
    # # Precompute permanent plane-to-plane edges
    # permanent_edges = compute_permanent_connectivity(global_prm, planes)
    
    # # Let the system generate candidates, or set to {} to force BFS to compute from geometry
    # user_candidates = {
    # 'P1-P2': [['B1', 'P1']],
    # 'P2-P3': [['B2', 'P2']],}
    
    # # --- Goal on Plane 2 (center) ---
    # # goal_xyz = (10,15, 3)
    # goal_xyz = (2,15, 5)
    
    # #Visualize the initial environment.
    # # visualize_global_prm(global_prm, planes, blocks_list, title="Initial Environment with Global PRM")
    
    # # --- Valid gaps ---
    # valid_gaps = [(1, 2),(2, 3)]
    
    # # --- Plan ---
    # planner = Planner(global_prm,
    #                   [local_ground, local_p1, local_p2],
    #                   blocks_list,
    #                   robot,
    #                   goal=goal_xyz,
    #                   goal_threshold=1.5,
    #                   valid_gaps=valid_gaps,
    #                   planes=planes,
    #                   permanent_edges=permanent_edges,
    #                   user_candidates=user_candidates)
    
    # plan_actions, reach_timeline = planner.plan(return_reachability=True)
    # print("plan_actions =", plan_actions)
    
    # # --- Visualization (for paper images) ---
    # viz.visualize_bbnamo(planes, blocks_list, plan_actions, goal_xyz,
    #                      limits=((-0.01,20),(-0.01,20),(0,6)), pause_dt=0.25,
    #                      reachable_plane_names_timeline=reach_timeline)


    """============================
    Paper figure: Figures 4
    ============================"""
    
    # # --- Planes ---
    # ground_vertices = [(-2, 0, 0), (30, 0, 0), (30, 30, 0), (-2, 30, 0)]
    # table1_vertices = [(2, 8, 2), (9, 8, 2), (9, 20, 2), (2, 20, 2)]
    # table2_vertices = [(15.5, 11, 3), (29, 11, 3), (29, 24, 3), (15.5, 24, 3)]
    # O_vertices = [(5, 14, 3), (12, 14, 3), (12, 18, 3), (5, 18, 3)]

    
    
    
    # ground_plane = Plane("Ground", ground_vertices)
    # table1_plane = Plane("Table1", table1_vertices)
    # table2_plane = Plane("Table2", table2_vertices)
    # O_plane = Plane("O", O_vertices)
    # planes = [ground_plane, table1_plane, table2_plane, O_plane]
    # add_higher_plane_obstacles(planes)
     
    
    
    # # Create the block and the robot. # Create two blocks.
    # block1 = Block(centroid=(16, 3, 0), length=2, width=2, height=1, color="orange", name="B1")
    # block2 = Block(centroid=(20, 6, 0), length=2, width=2, height=3, color="red", name="B2")
    
    # blocks_list = [block1,block2]
    
    # robot = Robot(position=(0, 0, 0), jump_height=1.2, horizontal_range=3, move_range=3, reach_range=1.9)
    
    # # Build local PRMs.
    # local_ground = LocalPRM(ground_plane, connection_threshold=2.0, all_blocks=blocks_list)
    # local_table1 = LocalPRM(table1_plane, connection_threshold=2.0, all_blocks=blocks_list)
    # local_table2 = LocalPRM(table2_plane, connection_threshold=2.0, all_blocks=blocks_list)
    # local_O = LocalPRM(O_plane, connection_threshold=2.0, all_blocks=blocks_list)
    
    # # Sample nodes on each plane.
    # local_ground.sample_nodes(550)
    # local_ground.add_node((0, 0, 0),meta={"source": local_ground.plane.name, "source_type": "Plane", "blocks": blocks_list})
    # local_table1.sample_nodes(50)
    # local_table2.sample_nodes(50)
    # local_O.sample_nodes(10)
    
    # # Connect nodes in each local PRM.
    # local_ground.connect_nodes()
    # local_table1.connect_nodes()
    # local_table2.connect_nodes()
    # local_O.connect_nodes()
    
    # # Build the global PRM.
    # global_prm = GlobalPRM(jump_threshold=1.2, horizontal_range=3, block_extra_factor=1.5)
    # global_prm.build_from_local([local_ground, local_table1, local_table2, local_O], blocks_list)
    
    # permanent_edges = compute_permanent_connectivity(global_prm, planes)
    # end_time = time.perf_counter()
    # print(f"Global PRM built in {end_time - start_time:.4f} seconds")
    
    # # Visualize the initial environment.
    # # visualize_global_prm(global_prm, planes, blocks_list, title="Initial Environment with Global PRM")
    
    
    # # Example usage:
    # start_time2 = time.perf_counter()
    # # For valid_gaps, assume you use find_valid_gaps(planes, dgap) with an appropriate dgap.
    # # valid_gaps = find_valid_gaps(planes, dgap=5.0)
    # valid_gaps=[(0,1),(0,2),(3,2)]
    # # user_candidates = None
    # user_candidates = {
    # 'Ground-Table1': [['B1', 'Ground']], 
    # 'O-Table2': [['B2', 'Ground']], 
    # # 'O-Y': [['B3', 'Table1']],
    # }
    # goal=(24,20,3)
    # planner = Planner(global_prm,  [local_ground, local_table1, local_table2, local_O], blocks_list, robot, goal, goal_threshold=2.0, valid_gaps=valid_gaps, planes=planes, permanent_edges=permanent_edges, user_candidates=user_candidates)
    # # planner = Planner(global_prm,  [local_ground, local_table1, local_table2, local_O, local_Y, local_R, local_P], blocks_list, robot, goal=(20,19,4), goal_threshold=2.0, valid_gaps=valid_gaps, planes=planes)
    # plan_actions, reach_timeline = planner.plan(return_reachability=True)
    # print("plan_actions =", plan_actions)
    
    # # --- Visualization (for paper images) ---
    # viz.visualize_bbnamo(planes, blocks_list, plan_actions, goal,
    #                      limits=((-2,30),(0,30),(0,10)), pause_dt=0.25,
    #                      reachable_plane_names_timeline=reach_timeline)
    
    
    
    





    
    """ video 1 horizon """    
    # ground_vertices = [(-2, 0, 0), (30, 0, 0), (30, 30, 0), (-2, 30, 0)]
    # table1_vertices = [(2, 8, 2), (9, 8, 2), (9, 20, 2), (2, 20, 2)]
    # table2_vertices = [(15.5, 11, 3), (29, 11, 3), (29, 24, 3), (15.5, 24, 3)]
    # O_vertices = [(5, 14, 3), (12, 14, 3), (12, 18, 3), (5, 18, 3)]
    # # Y_vertices = [(2, 14, 5), (6, 14, 5), (6, 18, 5), (2, 18, 5)]
    # # R_vertices = [(20, 18, 4), (25, 18, 4), (25, 24, 4), (20, 24, 4)]
    
    # # ground_vertices = [(-2, 0, 0), (24, 0, 0), (24, 20, 0), (-2, 20, 0)]
    # # table1_vertices = [(2, 4, 2), (12, 4, 2), (12, 11, 2), (2, 11, 2)]
    # # table2_vertices = [(17, 7, 2), (23, 7, 2), (23, 13, 2), (17, 13, 2)]
    # # O_vertices = [(9, 8, 3), (12, 8, 3), (12, 11, 3), (9, 11, 3)]
    # # Y_vertices = [(2, 7, 5), (6, 7, 5), (6, 11, 5), (2, 11, 5)]
    # # R_vertices = [(17, 10, 4), (20, 10, 4), (20, 13, 4), (17, 13, 4)]
    # # P_vertices = [(17, 17, 4), (23, 17, 4), (23, 20, 4), (17, 20, 4)]

    
    
    
    # ground_plane = Plane("Ground", ground_vertices)
    # table1_plane = Plane("Table1", table1_vertices)
    # table2_plane = Plane("Table2", table2_vertices)
    # O_plane = Plane("O", O_vertices)
    # # Y_plane = Plane("Y", Y_vertices)
    # # R_plane = Plane("R", R_vertices)
    # # P_plane = Plane("P", P_vertices)
    # planes = [ground_plane, table1_plane, table2_plane, O_plane]
    # # planes = [ground_plane, table1_plane, table2_plane, O_plane, Y_plane, R_plane, P_plane]
    # add_higher_plane_obstacles(planes)
     
    # # (Optional) Add higher plane obstacles if needed.
    # # add_higher_plane_obstacles([ground_plane, table1_plane, table2_plane])
    
    
    # # Create the block and the robot. # Create two blocks.
    # block1 = Block(centroid=(16, 3, 0), length=2, width=2, height=1, color="orange", name="B1")
    # block2 = Block(centroid=(20, 6, 0), length=2, width=2, height=3, color="red", name="B2")
    # # block3 = Block(centroid=(21, 23, 4), length=2, width=2, height=2, color="green", name="B3")
    
    # # Create the block and the robot. # Create two blocks.
    # # block1 = Block(centroid=(18, 3, 0), length=2, width=2, height=1, color="red")
    # # block2 = Block(centroid=(3, 5, 2), length=2, width=2, height=1, color="red")
    # # block3 = Block(centroid=(18.5, 11.5, 4), length=2, width=2, height=2, color="green")
    # # block4 = Block(centroid=(3, 9, 5), length=2, width=2, height=3, color="blue")
    # blocks_list = [block1,block2]
    # # blocks_list = [block1,block2,block3,block4]
    
    # robot = Robot(position=(0, 0, 0), jump_height=1.2, horizontal_range=3, move_range=3, reach_range=1.9)
    
    # # Build local PRMs.
    # local_ground = LocalPRM(ground_plane, connection_threshold=2.0, all_blocks=blocks_list)
    # local_table1 = LocalPRM(table1_plane, connection_threshold=2.0, all_blocks=blocks_list)
    # local_table2 = LocalPRM(table2_plane, connection_threshold=2.0, all_blocks=blocks_list)
    # local_O = LocalPRM(O_plane, connection_threshold=2.0, all_blocks=blocks_list)
    # # local_Y = LocalPRM(Y_plane, connection_threshold=2.0, all_blocks=blocks_list)
    # # local_R = LocalPRM(R_plane, connection_threshold=2.0, all_blocks=blocks_list)
    # # local_P = LocalPRM(P_plane, connection_threshold=2.0, all_blocks=blocks_list)
    
    # # Sample nodes on each plane.
    # local_ground.sample_nodes(350)
    # local_ground.add_node((0, 0, 0),meta={"source": local_ground.plane.name, "source_type": "Plane", "blocks": blocks_list})
    # local_table1.sample_nodes(50)
    # local_table2.sample_nodes(50)
    # local_O.sample_nodes(10)
    # # local_Y.sample_nodes(10)
    # # local_R.sample_nodes(20)
    # # local_P.sample_nodes(15)
    
    # # Connect nodes in each local PRM.
    # local_ground.connect_nodes()
    # local_table1.connect_nodes()
    # local_table2.connect_nodes()
    # local_O.connect_nodes()
    # # local_Y.connect_nodes()
    # # local_R.connect_nodes()
    # # local_P.connect_nodes()
    
    # # Build the global PRM.
    # global_prm = GlobalPRM(jump_threshold=1.2, horizontal_range=3, block_extra_factor=1.5)
    # global_prm.build_from_local([local_ground, local_table1, local_table2, local_O], blocks_list)
    # # global_prm.build_from_local([local_ground, local_table1, local_table2, local_O, local_Y, local_R, local_P], blocks_list)
    
    
    
    # permanent_edges = compute_permanent_connectivity(global_prm, planes)

    
    # end_time = time.perf_counter()
    # print(f"Global PRM built in {end_time - start_time:.4f} seconds")
    
    # # Visualize the initial environment.
    # visualize_global_prm(global_prm, planes, blocks_list, title="Initial Environment with Global PRM")
    
    
    
    # # Example usage:
    # start_time2 = time.perf_counter()
    # # For valid_gaps, assume you use find_valid_gaps(planes, dgap) with an appropriate dgap.
    # # valid_gaps = find_valid_gaps(planes, dgap=5.0)
    # valid_gaps=[(0,1),(0,2),(3,4)]
    # # user_candidates = None
    # user_candidates = {
    # 'Ground-Table1': [['B1', 'Ground']], 
    # 'O-Table2': [['B2', 'Ground']], 
    # # 'O-Y': [['B3', 'Table1']],
    # }
    
    # planner = Planner(global_prm,  [local_ground, local_table1, local_table2, local_O], blocks_list, robot, goal=(24,20,3), goal_threshold=2.0, valid_gaps=valid_gaps, planes=planes, permanent_edges=permanent_edges, user_candidates=user_candidates)
    # # planner = Planner(global_prm,  [local_ground, local_table1, local_table2, local_O, local_Y, local_R, local_P], blocks_list, robot, goal=(20,19,4), goal_threshold=2.0, valid_gaps=valid_gaps, planes=planes)
    # plan_actions = planner.plan()
    # end_time = time.perf_counter()
    # print(f"Plan found in {end_time - start_time:.4f} seconds")
    # #  print("plan_actions=", plan_actions)
    
    # print("BFS_counter=", BFS_counter)
    # print("number of nodes", len(planner.tree))
    
    # # viz.visualize_bbnamo(planes, blocks_list, plan_actions,
    # #                      limits=((-2,30),(0,30),(0,10)), pause_dt=4.15)















