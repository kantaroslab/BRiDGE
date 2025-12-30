#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Jul 19 12:16:12 2025

@author: samarth
"""

import matplotlib
# Use an interactive backend
matplotlib.use('tkagg')
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
import numpy as np

# Enable interactive mode for live updates and 3D rotation
# plt.ion()

# -- Visualization primitives ------------------------------------------------

# def draw_solid_block(ax, corners, height, color="cyan", alpha=0.6):
#     """
#     Draw a filled plane from z=0 up to `height` at the polygon footprint `corners`.
#     """
#     bottom = [[x, y, height-2] for x, y, _ in corners]
#     top    = [[x, y, height] for x, y, _ in corners]
#     faces = [
#         [bottom[0], bottom[1], top[1], top[0]],
#         [bottom[1], bottom[2], top[2], top[1]],
#         [bottom[2], bottom[3], top[3], top[2]],
#         [bottom[3], bottom[0], top[0], top[3]],
#         top,
#         bottom
#     ]
#     for f in faces:
#         poly = Poly3DCollection([f], color=color, alpha=alpha, edgecolor='k')
#         ax.add_collection3d(poly)

def draw_solid_block(ax, corners, height,bottom, color="cyan", alpha=0.6):
    """
    Extrude the polygon footprint up from z=0 to `height`.
    """
    bottom = [[x, y, bottom] for x, y, _ in corners]
    top    = [[x, y, height] for x, y, _ in corners]
    faces = [
        [bottom[0], bottom[1], top[1], top[0]],
        [bottom[1], bottom[2], top[2], top[1]],
        [bottom[2], bottom[3], top[3], top[2]],
        [bottom[3], bottom[0], top[0], top[3]],
        top,
        bottom
    ]
    for f in faces:
        poly = Poly3DCollection([f], facecolor=color, alpha=alpha, edgecolor='k')
        ax.add_collection3d(poly)

def draw_block(ax, position, dimensions, color="orange", alpha=1):
    """
    Draw a cuboid block centered at `position` with (length, depth, height)=`dimensions`.
    """
    cx, cy, cz = position
    l, d, h     = dimensions
    x0, y0 = cx - l/2, cy - d/2
    v = [
        (x0    , y0    , cz),
        (x0 + l, y0    , cz),
        (x0 + l, y0 + d, cz),
        (x0    , y0 + d, cz)
    ]
    top = [(x, y, cz + h) for x, y, _ in v]
    faces = [v, top,
             [v[0], v[1], top[1], top[0]],
             [v[1], v[2], top[2], top[1]],
             [v[2], v[3], top[3], top[2]],
             [v[3], v[0], top[0], top[3]]]
    poly = Poly3DCollection(faces, facecolors=color, alpha=alpha, edgecolor='k')
    ax.add_collection3d(poly)

# def draw_reach_sphere(ax, center, radius, alpha=0.15):
#     cx, cy, cz = center
#     # coarse mesh is fine for a translucent cue
#     u = np.linspace(0, 2*np.pi, 24)
#     v = np.linspace(0, np.pi, 12)
#     xs = cx + radius*np.outer(np.cos(u), np.sin(v))
#     ys = cy + radius*np.outer(np.sin(u), np.sin(v))
#     zs = cz + radius*np.outer(np.ones_like(u), np.cos(v))
#     ax.plot_surface(xs, ys, zs, linewidth=0, antialiased=False, alpha=alpha)
def draw_reach_sphere(ax, center, radius_xy=3.0, radius_z=1.5, alpha=0.15):
    cx, cy, cz = center
    u = np.linspace(0, 2*np.pi, 30)
    v = np.linspace(0, np.pi, 15)
    xs = cx + radius_xy * np.outer(np.cos(u), np.sin(v))
    ys = cy + radius_xy * np.outer(np.sin(u), np.sin(v))
    zs = cz + radius_z  * np.outer(np.ones_like(u), np.cos(v))
    ax.plot_surface(xs, ys, zs, linewidth=0, antialiased=False, alpha=alpha)
    
def draw_lines(ax, lines, lw=2.0, alpha=0.8):
    for (x1,y1,z1), (x2,y2,z2) in lines:
        ax.plot([x1,x2], [y1,y2], [z1,z2], linewidth=lw, alpha=alpha)
from mpl_toolkits.mplot3d.art3d import Line3DCollection

def draw_lines_collection(ax, lines, color='magenta', lw=2.0, alpha=1):
    if not lines:
        return None
    segs = [[(x1,y1,z1), (x2,y2,z2)] for (x1,y1,z1),(x2,y2,z2) in lines]
    lc = Line3DCollection(segs, linewidths=lw, alpha=alpha, colors=color)
    ax.add_collection3d(lc)
    return lc

def redraw(ax, robot_pos, block_info_list, env_planes, limits, goal,
           reach_sphere=None, connectivity_lines=None, reachable_plane_idxs=None):
    ax.cla()

    planes_as_blocks = False
    reachable_plane_idxs = set() if reachable_plane_idxs is None else set(reachable_plane_idxs)
    # Dynamic plane styling (index 0 is typically Ground)

    # planes
    for ct, (corners, height) in enumerate(env_planes):
        if planes_as_blocks:
            # if you use this path, keep your bottom[] logic consistent
            pass
        else:
            top_face = [(x, y, height) for x, y, _ in corners]
            poly = Poly3DCollection([top_face],
                                    facecolor=('lightgreen' if (ct in reachable_plane_idxs and ct != 0) else ('white' if ct == 0 else 'teal')),
                                    alpha=(0.55 if (ct in reachable_plane_idxs and ct != 0) else (0.0 if ct == 0 else 0.30)),
                                    edgecolor='k')
            ax.add_collection3d(poly)

    # blocks
    for b in block_info_list:
        draw_block(ax, b['position'], b['dimensions'], color=b.get('color','orange'))

    # robot
    draw_block(ax, robot_pos, (1, 1, 0.25), color='black', alpha=1.0)
    
    xg, yg, zg =  goal  # tuple (x,y,z)
    # xg, yg, zg = (4,16,5)  # tuple (x,y,z)
    ax.scatter([xg], [yg], [zg],
               marker='*', s=300, c='gold',
               edgecolors='k')

    # overlays — draw after all opaque geometry
    if connectivity_lines:
        draw_lines_collection(ax, connectivity_lines, lw=2.5, alpha=0.95)
    if reach_sphere:
        draw_reach_sphere(ax, reach_sphere["center"], reach_sphere["radius"],  reach_sphere["radiusZ"], alpha=0.18)

    # axes + single final draw
    (xlim, ylim, zlim) = limits
    ax.set_xlim(*xlim); ax.set_ylim(*ylim); ax.set_zlim(*zlim)
    ax.set_xlabel('X'); ax.set_ylabel('Y'); ax.set_zlabel('Z')
    fig = ax.get_figure()
    fig.canvas.draw()
    fig.canvas.flush_events()

def animate_plan(plan_actions, env_planes, block_info_list, goal, limits=((-2,30),(0,30),(0,6)), pause_dt=2, reachable_timeline=None):
    """
    Step through `plan_actions` and animate robot + blocks interactively.
    """
    # 1) define where to show the reach sphere (global waypoint indices)
    sphere_wp_idxs = {#34,35,36, 39,40, 41, 42
                      }   # whatever indices best illustrate “can’t reach without the box”
    reach_XY = 3
    reach_Z = 1.5             # your hard-coded max single-step distance
    
    # 2) define all possible plane-connectivity lines (once)
    all_lines = [
    ]
    
    # 3) schedule which subset to show as connectivity evolves
    lines_at = [
        ]
    
    # 4) when to advance to each subset (by waypoint index)
    line_switch_wp_idxs = [
        # 0, 13, 29
        ]  # must be non-decreasing; len == len(lines_at)
    
    sticky_reach = None
    
    sphere_wp_idxs =  set(sphere_wp_idxs or [])
    all_lines = all_lines or []
    lines_at = lines_at or [[]]
    line_switch_wp_idxs = list(line_switch_wp_idxs or [])
    curr_lineset_idx = 0
    wp_counter = 0
    
    def current_lines():
        if curr_lineset_idx < len(lines_at):
            idxs = lines_at[curr_lineset_idx]
            return [all_lines[i] for i in idxs]
        return []

    def current_sphere():
        if reach_XY and (wp_counter in sphere_wp_idxs):
            return {"center": robot_pos, "radius": float(reach_XY), "radiusZ": float(reach_Z)}
        return None

    
    fig = plt.figure()
    ax  = fig.add_subplot(111, projection='3d')
    # Show window
    plt.show(block=False)  # display non-blocking window
    # window activation not needed for TkAgg backend

    robot_pos = (0,0,0)
    for action, data in plan_actions:
        if action == 'GoTo' and len(data) > 0:
            robot_pos = tuple(data[0])
            break
    picked_idx = None
    # initial draw
    sticky_reach = reachable_timeline[0] if (reachable_timeline and len(reachable_timeline) > 0) else None

    init_reach = sticky_reach
    redraw(ax, robot_pos, block_info_list, env_planes, limits, goal,
           reach_sphere=current_sphere(),
           connectivity_lines=current_lines(),
           reachable_plane_idxs=init_reach)
    plt.pause(11.5)

    for action_idx, (action, data) in enumerate(plan_actions):
        if reachable_timeline is not None and action_idx < len(reachable_timeline):
            sticky_reach = reachable_timeline[action_idx]
        curr_reach = sticky_reach
        if action == 'GoTo':
            for wp in data:
                robot_pos = tuple(wp)
                if picked_idx is not None:
                    x, y, z = robot_pos
                    h = float(block_info_list[picked_idx]['dimensions'][2])
                    block_info_list[picked_idx]['position'] = (x, y, z + 0.5)
                # update lineset if needed
                while line_switch_wp_idxs and curr_lineset_idx+1 < len(lines_at) \
                        and wp_counter >= line_switch_wp_idxs[curr_lineset_idx+1]:
                    curr_lineset_idx += 1

                redraw(ax, robot_pos, block_info_list, env_planes, limits, goal,
                   reach_sphere=current_sphere(),
                   connectivity_lines=current_lines(),
                   reachable_plane_idxs=curr_reach)
                plt.pause(pause_dt)
                # advance global waypoint counter AFTER drawing
                wp_counter += 1

        elif action == 'Pick':
            picked_idx = data
            # show connectivity update immediately after pick
            redraw(ax, robot_pos, block_info_list, env_planes, limits, goal,
                   reach_sphere=current_sphere(),
                   connectivity_lines=current_lines(),
                   reachable_plane_idxs=curr_reach)
            plt.pause(1)

        elif action == 'Drop':
            block_info_list[picked_idx]['position'] = tuple(data)
            picked_idx = None
            redraw(ax, robot_pos, block_info_list, env_planes, limits, goal, reachable_plane_idxs=curr_reach)
            plt.pause(1)

    # Keep window open
    plt.ioff()
    plt.show()

# ---- BB_NAMO integration helpers -------------------------------------------

def _to_float_tuple(p):
    # Accepts (np.float64, np.float64, np.float64) or plain floats
    return (float(p[0]), float(p[1]), float(p[2]))

def _sanitize_plan_actions(plan_actions):
    """
    Convert any np.float64 tuples to native floats, and ensure the structure is:
      - ('GoTo', [ (x,y,z), ... ])
      - ('Pick', int_block_index)
      - ('Drop', (x,y,z))
    """
    out = []
    for action_idx, (action, data) in enumerate(plan_actions):
        if action == 'GoTo':
            waypoints = [_to_float_tuple(wp) for wp in data]
            out.append((action, waypoints))
        elif action == 'Pick':
            out.append((action, int(data)))
        elif action == 'Drop':
            out.append((action, _to_float_tuple(data)))
        else:
            # Tolerate unknown actions by skipping them (or raise if you prefer)
            continue
    return out

def bbnamo_to_visualizer_inputs(planes, blocks, plan_actions):
    """
    Build (env_planes, block_info_list, sanitized_plan_actions) from BB_NAMO objects.
      - planes: list of Plane objects (with .vertices)
      - blocks: list of Block objects (with .centroid, .length, .width, .height, .color)
      - plan_actions: list of ('GoTo'/'Pick'/'Drop', payload) from BB_NAMO
    """
    # env_planes: [([ (x,y,z), ... ], z_level), ...]
    env_planes = []
    for pl in planes:
        corners = [tuple(v) for v in pl.vertices]  # keep as (x,y,z)
        z_level = corners[0][2] if corners else 0.0
        env_planes.append((corners, float(z_level)))

    # block_info_list: [{'dimensions':(L,W,H), 'position':(x,y,z), 'color':...}, ...]
    block_info_list = []
    for b in blocks:
        dims = (float(b.length), float(b.width), float(b.height))
        pos  = (float(b.centroid[0]), float(b.centroid[1]), float(b.centroid[2]))
        color = getattr(b, "color", "orange")
        block_info_list.append({'dimensions': dims, 'position': pos, 'color': color})

    sanitized = _sanitize_plan_actions(plan_actions)
    return env_planes, block_info_list, sanitized

def visualize_bbnamo(planes, blocks, plan_actions,goal,
                     limits=((-2,30),(0,30),(0,6)), pause_dt=0.2,
                     reachable_plane_names_timeline=None):
    """
    One-call integration from BB_NAMO to the visualizer.
    """
    env_planes, block_info_list, sanitized = bbnamo_to_visualizer_inputs(
        planes, blocks, plan_actions
    )
    reachable_timeline = None
    if reachable_plane_names_timeline is not None:
        # convert plane-name sets -> plane-index sets consistent with `planes` ordering
        name_to_idx = {getattr(p,'name',str(i)): i for i, p in enumerate(planes)}
        reachable_timeline = []
        for s in reachable_plane_names_timeline:
            if s is None:
                reachable_timeline.append(set())
            else:
                reachable_timeline.append({name_to_idx[n] for n in s if n in name_to_idx})
    animate_plan(sanitized, env_planes, block_info_list, goal, limits=limits, pause_dt=pause_dt,
                 reachable_timeline=reachable_timeline)


def _overlay_global_prm(ax3d, global_prm, node_size=6, edge_lw=0.5, alpha=0.35):
    """
    Draw the Global PRM (nodes + edges) onto an existing 3D axes.
    - ax3d: matplotlib 3D axes (projection='3d')
    - global_prm: object with attributes .nx_graph (NetworkX) and .nodes (indexable -> (x,y,z))
    """
    G = getattr(global_prm, "nx_graph", None)
    pts = getattr(global_prm, "nodes", None)
    if G is None or pts is None:
        return

    # Edges
    for u, v in G.edges():
        xu, yu, zu = pts[u]
        xv, yv, zv = pts[v]
        ax3d.plot([xu, xv], [yu, yv], [zu, zv], linewidth=edge_lw, alpha=alpha)

    # Nodes
    xs, ys, zs = zip(*pts)
    ax3d.scatter(xs, ys, zs, s=node_size, alpha=min(1.0, alpha + 0.25))


    
    
    
    
    
    
if __name__ == '__main__':
    # environment planes
    env_planes = [
        ([(-2, 0, 0), (30, 0, 0), (30, 30, 0), (-2, 30, 0)], 0),
        ([(2, 8, 2), (12, 8, 2), (12, 20, 2), (2, 20, 2)], 2),
        ([(20, 11, 2), (29, 11, 2), (29, 24, 2), (20, 24, 2)], 2),
        ([(9, 14, 3), (12, 14, 3), (12, 18, 3), (9, 18, 3)], 3),
        ([(2, 14, 5), (6, 14, 5), (6, 18, 5), (2, 18, 5)], 5),
        ([(20, 18, 4), (25, 18, 4), (25, 24, 4), (20, 24, 4)], 4)
    ]

    # initial block info
    block_info_list = [
        {'dimensions': (2, 2, 1), 'position': (18, 3, 0), 'color': 'orange'},
        {'dimensions': (2, 2, 1), 'position': (8, 11, 2), 'color': 'red'},
        {'dimensions': (2, 2, 2), 'position': (21, 23, 4), 'color': 'green'}
    ]

    # full plan_actions list here...
    # plan_actions= [('GoTo', [(np.float64(-0.30601816050710084), np.float64(0.30830678605541895), np.float64(0.0)), (np.float64(1.8303866363671695), np.float64(0.24701501938919845), np.float64(0.0)), (np.float64(3.0817120449787154), np.float64(0.05281121982409598), np.float64(0.0)), (np.float64(4.894302256024357), np.float64(0.058554137103200565), np.float64(0.0)), (np.float64(7.568883000815394), np.float64(0.4888356337937305), np.float64(0.0)), (np.float64(10.330751831110057), np.float64(0.542046554224489), np.float64(0.0)), (np.float64(13.121140243545426), np.float64(1.3466978210469016), np.float64(0.0)), (np.float64(14.195073363920702), np.float64(1.4702827994813805), np.float64(0.0)), (np.float64(18.0), np.float64(3.0), np.float64(1.0)), (np.float64(17.061478363425817), np.float64(3.0868610518378503), np.float64(0.0))]), ('Pick', 0), ('GoTo', [(np.float64(17.061478363425817), np.float64(3.0868610518378503), np.float64(0.0)), (np.float64(18.0), np.float64(3.0), np.float64(1.0)), (np.float64(13.933486921084995), np.float64(3.3501515055696007), np.float64(0.0)), (np.float64(11.620232034471417), np.float64(4.15273045694868), np.float64(0.0)), (np.float64(10.524481973449621), np.float64(4.523523104377739), np.float64(0.0)), (np.float64(7.809667802456126), np.float64(5.336820244367776), np.float64(0.0)), (np.float64(6.673604591156412), np.float64(5.5478663852581445), np.float64(0.0)), (np.float64(4.205064415385415), np.float64(5.80974386475026), np.float64(0.0)), (np.float64(2.6744775435071055), np.float64(5.937052935076132), np.float64(0.0)), (np.float64(0.6961445590902962), np.float64(6.38638368037586), np.float64(0.0)), (np.float64(-0.5610244285940524), np.float64(6.758557354334687), np.float64(0.0)), (np.float64(0.18533035236431772), np.float64(9.48416107869415), np.float64(0.0)), (np.float64(1.733318540957626), np.float64(11.813112012339898), np.float64(0.0)), (np.float64(1.5221295670116604), np.float64(13.069239070703043), np.float64(0.0)), (np.float64(-0.13141479267040168), np.float64(14.636332487901434), np.float64(0.0))]), ('Drop', (np.float64(0.8), np.float64(15.667038358692167), np.float64(0.0))), ('GoTo', [(np.float64(-0.13141479267040168), np.float64(14.636332487901434), np.float64(0.0)), (np.float64(0.8), np.float64(15.667038358692167), np.float64(1.0)), (np.float64(4.138383872359501), np.float64(12.669963359068369), np.float64(2.0)), (np.float64(8.0), np.float64(11.0), np.float64(3.0)), (np.float64(9.181791729429364), np.float64(10.383792655901019), np.float64(2.0))]), ('Pick', 1), ('GoTo', [(np.float64(9.181791729429364), np.float64(10.383792655901019), np.float64(2.0)), (np.float64(8.0), np.float64(11.0), np.float64(3.0)), (np.float64(4.138383872359501), np.float64(12.669963359068369), np.float64(2.0)), (np.float64(0.8), np.float64(15.667038358692167), np.float64(1.0)), (np.float64(1.7556904054041524), np.float64(19.633694620398124), np.float64(0.0)), (np.float64(3.4933863949017088), np.float64(20.708861195858578), np.float64(0.0)), (np.float64(5.996654256919445), np.float64(21.25797517848121), np.float64(0.0)), (np.float64(7.643284323684682), np.float64(21.389458742512527), np.float64(0.0)), (np.float64(10.262550932881421), np.float64(20.472690694992643), np.float64(0.0)), (np.float64(10.787633794225332), np.float64(20.388207304802688), np.float64(0.0)), (np.float64(13.515100151982352), np.float64(19.911499207665297), np.float64(0.0)), (np.float64(15.80894271642271), np.float64(18.20698834948443), np.float64(0.0)), (np.float64(17.567322457325766), np.float64(17.04614632751086), np.float64(0.0)), (np.float64(19.809959442021594), np.float64(15.158128560266572), np.float64(0.0))]), ('Drop', (np.float64(18.8), np.float64(15.033747339504576), np.float64(0.0))), ('GoTo', [(np.float64(19.809959442021594), np.float64(15.158128560266572), np.float64(0.0)), (np.float64(17.567322457325766), np.float64(17.04614632751086), np.float64(0.0)), (np.float64(15.80894271642271), np.float64(18.20698834948443), np.float64(0.0)), (np.float64(13.515100151982352), np.float64(19.911499207665297), np.float64(0.0)), (np.float64(10.787633794225332), np.float64(20.388207304802688), np.float64(0.0)), (np.float64(10.262550932881421), np.float64(20.472690694992643), np.float64(0.0)), (np.float64(7.643284323684682), np.float64(21.389458742512527), np.float64(0.0)), (np.float64(5.996654256919445), np.float64(21.25797517848121), np.float64(0.0)), (np.float64(3.4933863949017088), np.float64(20.708861195858578), np.float64(0.0)), (np.float64(1.7556904054041524), np.float64(19.633694620398124), np.float64(0.0)), (np.float64(0.8), np.float64(15.667038358692167), np.float64(1.0)), (np.float64(-0.13141479267040168), np.float64(14.636332487901434), np.float64(0.0))]), ('Pick', 0), ('GoTo', [(np.float64(-0.13141479267040168), np.float64(14.636332487901434), np.float64(0.0)), (np.float64(0.8), np.float64(15.667038358692167), np.float64(1.0)), (np.float64(1.7556904054041524), np.float64(19.633694620398124), np.float64(0.0)), (np.float64(3.4933863949017088), np.float64(20.708861195858578), np.float64(0.0)), (np.float64(5.996654256919445), np.float64(21.25797517848121), np.float64(0.0)), (np.float64(7.643284323684682), np.float64(21.389458742512527), np.float64(0.0)), (np.float64(10.262550932881421), np.float64(20.472690694992643), np.float64(0.0)), (np.float64(10.787633794225332), np.float64(20.388207304802688), np.float64(0.0)), (np.float64(13.515100151982352), np.float64(19.911499207665297), np.float64(0.0)), (np.float64(15.80894271642271), np.float64(18.20698834948443), np.float64(0.0)), (np.float64(18.8), np.float64(15.033747339504576), np.float64(1.0)), (np.float64(22.34931171265168), np.float64(15.394576545193082), np.float64(2.0))]), ('Drop', (np.float64(22.093203435574466), np.float64(16.8), np.float64(2.0))), ('GoTo', [(np.float64(22.34931171265168), np.float64(15.394576545193082), np.float64(2.0)), (np.float64(22.093203435574466), np.float64(16.8), np.float64(3.0)), (np.float64(22.601130191605726), np.float64(18.690906073809735), np.float64(4.0)), (np.float64(23.770730878440425), np.float64(21.297387178028487), np.float64(4.0)), (np.float64(22.23785827525131), np.float64(23.757876600232592), np.float64(4.0))]), ('Pick', 2), ('GoTo', [(np.float64(22.23785827525131), np.float64(23.757876600232592), np.float64(4.0)), (np.float64(23.770730878440425), np.float64(21.297387178028487), np.float64(4.0)), (np.float64(22.601130191605726), np.float64(18.690906073809735), np.float64(4.0)), (np.float64(22.093203435574466), np.float64(16.8), np.float64(3.0)), (np.float64(21.0524090659988), np.float64(14.987410914129672), np.float64(2.0)), (np.float64(18.8), np.float64(15.033747339504576), np.float64(1.0)), (np.float64(14.622701970030644), np.float64(15.527738049164805), np.float64(0.0)), (np.float64(12.651418487071993), np.float64(16.84129304057658), np.float64(0.0))]), ('Drop', (np.float64(13.2), np.float64(16.23092211187797), np.float64(0.0))), ('GoTo', [(np.float64(12.651418487071993), np.float64(16.84129304057658), np.float64(0.0)), (np.float64(14.622701970030644), np.float64(15.527738049164805), np.float64(0.0)), (np.float64(17.081150891255067), np.float64(14.761119683936455), np.float64(0.0)), (np.float64(17.751066914068417), np.float64(15.080536653483023), np.float64(0.0))]), ('Pick', 1), ('GoTo', [(np.float64(17.751066914068417), np.float64(15.080536653483023), np.float64(0.0)), (np.float64(17.567322457325766), np.float64(17.04614632751086), np.float64(0.0)), (np.float64(15.80894271642271), np.float64(18.20698834948443), np.float64(0.0)), (np.float64(13.515100151982352), np.float64(19.911499207665297), np.float64(0.0)), (np.float64(10.787633794225332), np.float64(20.388207304802688), np.float64(0.0)), (np.float64(10.262550932881421), np.float64(20.472690694992643), np.float64(0.0)), (np.float64(7.643284323684682), np.float64(21.389458742512527), np.float64(0.0)), (np.float64(5.996654256919445), np.float64(21.25797517848121), np.float64(0.0)), (np.float64(3.4933863949017088), np.float64(20.708861195858578), np.float64(0.0)), (np.float64(1.7556904054041524), np.float64(19.633694620398124), np.float64(0.0)), (np.float64(0.5955055747967783), np.float64(18.907960092384638), np.float64(0.0)), (np.float64(-0.6178826528017431), np.float64(16.203894567747735), np.float64(0.0)), (np.float64(-0.13141479267040168), np.float64(14.636332487901434), np.float64(0.0)), (np.float64(1.5221295670116604), np.float64(13.069239070703043), np.float64(0.0))]), ('Drop', (np.float64(0.8), np.float64(13.225798737001313), np.float64(0.0))), ('GoTo', [(np.float64(1.5221295670116604), np.float64(13.069239070703043), np.float64(0.0)), (np.float64(-0.13141479267040168), np.float64(14.636332487901434), np.float64(0.0)), (np.float64(-0.6178826528017431), np.float64(16.203894567747735), np.float64(0.0)), (np.float64(0.5955055747967783), np.float64(18.907960092384638), np.float64(0.0)), (np.float64(1.7556904054041524), np.float64(19.633694620398124), np.float64(0.0)), (np.float64(3.4933863949017088), np.float64(20.708861195858578), np.float64(0.0)), (np.float64(5.996654256919445), np.float64(21.25797517848121), np.float64(0.0)), (np.float64(7.643284323684682), np.float64(21.389458742512527), np.float64(0.0)), (np.float64(10.262550932881421), np.float64(20.472690694992643), np.float64(0.0)), (np.float64(10.787633794225332), np.float64(20.388207304802688), np.float64(0.0)), (np.float64(13.515100151982352), np.float64(19.911499207665297), np.float64(0.0)), (np.float64(15.44201566849232), np.float64(18.366079952262684), np.float64(0.0)), (np.float64(14.622701970030644), np.float64(15.527738049164805), np.float64(0.0)), (np.float64(12.651418487071993), np.float64(16.84129304057658), np.float64(0.0))]), ('Pick', 2), ('GoTo', [(np.float64(12.651418487071993), np.float64(16.84129304057658), np.float64(0.0)), (np.float64(14.622701970030644), np.float64(15.527738049164805), np.float64(0.0)), (np.float64(15.44201566849232), np.float64(18.366079952262684), np.float64(0.0)), (np.float64(13.515100151982352), np.float64(19.911499207665297), np.float64(0.0)), (np.float64(10.787633794225332), np.float64(20.388207304802688), np.float64(0.0)), (np.float64(10.262550932881421), np.float64(20.472690694992643), np.float64(0.0)), (np.float64(7.643284323684682), np.float64(21.389458742512527), np.float64(0.0)), (np.float64(5.996654256919445), np.float64(21.25797517848121), np.float64(0.0)), (np.float64(3.4933863949017088), np.float64(20.708861195858578), np.float64(0.0)), (np.float64(1.7556904054041524), np.float64(19.633694620398124), np.float64(0.0)), (np.float64(0.5955055747967783), np.float64(18.907960092384638), np.float64(0.0)), (np.float64(-0.6178826528017431), np.float64(16.203894567747735), np.float64(0.0)), (np.float64(0.8), np.float64(13.225798737001313), np.float64(1.0)), (np.float64(4.850576287128186), np.float64(12.900083517227444), np.float64(2.0)), (np.float64(5.710733245203751), np.float64(13.699904126962089), np.float64(2.0)), (np.float64(6.841551040462982), np.float64(15.148362751419107), np.float64(2.0)), (np.float64(7.15532273259052), np.float64(16.682256502131413), np.float64(2.0))]), ('Drop', (np.float64(7.2), np.float64(15.740782553633101), np.float64(2.0))), ('GoTo', [(np.float64(7.15532273259052), np.float64(16.682256502131413), np.float64(2.0)), (np.float64(9.01867281796113), np.float64(15.492702894413931), np.float64(3.0)), (np.float64(7.2), np.float64(15.740782553633101), np.float64(4.0)), (np.float64(3.579457091470507), np.float64(16.85256082854877), np.float64(5.0))])]
    plan_actions = [('GoTo', [
        (np.float64(-0.30601816050710084), np.float64(0.30830678605541895), np.float64(0.0)),
        (np.float64(1.8303866363671695), np.float64(0.24701501938919845), np.float64(0.0)),
        (np.float64(3.0817120449787154), np.float64(0.05281121982409598), np.float64(0.0)),
        (np.float64(4.894302256024357), np.float64(0.058554137103200565), np.float64(0.0)),
        (np.float64(7.568883000815394), np.float64(0.4888356337937305), np.float64(0.0)),
        (np.float64(10.330751831110057), np.float64(0.542046554224489), np.float64(0.0)),
        (np.float64(13.121140243545426), np.float64(1.3466978210469016), np.float64(0.0)),
        (np.float64(14.195073363920702), np.float64(1.4702827994813805), np.float64(0.0)),
        # (np.float64(18.0), np.float64(3.0), np.float64(1.0)),
        (np.float64(17.061478363425817), np.float64(3.0868610518378503), np.float64(0.0)),
    ]),
    
    ('Pick', 0),
    
    ('GoTo', [
        (np.float64(17.061478363425817), np.float64(3.0868610518378503), np.float64(0.0)),
        # (np.float64(18.0), np.float64(3.0), np.float64(1.0)),
        (np.float64(13.933486921084995), np.float64(3.3501515055696007), np.float64(0.0)),
        (np.float64(11.620232034471417), np.float64(4.15273045694868), np.float64(0.0)),
        (np.float64(10.524481973449621), np.float64(4.523523104377739), np.float64(0.0)),
        (np.float64(7.809667802456126), np.float64(5.336820244367776), np.float64(0.0)),
        (np.float64(6.673604591156412), np.float64(5.5478663852581445), np.float64(0.0)),
        (np.float64(4.205064415385415), np.float64(5.80974386475026), np.float64(0.0)),
        (np.float64(2.6744775435071055), np.float64(5.937052935076132), np.float64(0.0)),
        (np.float64(0.6961445590902962), np.float64(6.38638368037586), np.float64(0.0)),
        (np.float64(-0.5610244285940524), np.float64(6.758557354334687), np.float64(0.0)),
        (np.float64(0.18533035236431772), np.float64(9.48416107869415), np.float64(0.0)),
        (np.float64(1.733318540957626), np.float64(11.813112012339898), np.float64(0.0)),
        (np.float64(1.5221295670116604), np.float64(13.069239070703043), np.float64(0.0)),
        (np.float64(-0.13141479267040168), np.float64(14.636332487901434), np.float64(0.0)),
    ]),
    
    ('Drop', (np.float64(0.8), np.float64(15.667038358692167), np.float64(0.0))),
    
    ('GoTo', [
        (np.float64(-0.13141479267040168), np.float64(14.636332487901434), np.float64(0.0)),
        (np.float64(0.8), np.float64(15.667038358692167), np.float64(1.0)),
        (np.float64(4.138383872359501), np.float64(12.669963359068369), np.float64(2.0)),
        (np.float64(8.0), np.float64(12.5), np.float64(2.0)),
        (np.float64(9.181791729429364), np.float64(10.383792655901019), np.float64(2.0)),
    ]),
    
    ('Pick', 1),
    
    ('GoTo', [
        (np.float64(9.181791729429364), np.float64(10.383792655901019), np.float64(2.0)),
        (np.float64(8.0), np.float64(12.5), np.float64(2.0)),
        (np.float64(4.138383872359501), np.float64(12.669963359068369), np.float64(2.0)),
        (np.float64(0.8), np.float64(15.667038358692167), np.float64(1.0)),
        (np.float64(1.7556904054041524), np.float64(19.633694620398124), np.float64(0.0)),
        (np.float64(3.4933863949017088), np.float64(20.708861195858578), np.float64(0.0)),
        (np.float64(5.996654256919445), np.float64(21.25797517848121), np.float64(0.0)),
        (np.float64(7.643284323684682), np.float64(21.389458742512527), np.float64(0.0)),
        (np.float64(10.262550932881421), np.float64(20.472690694992643), np.float64(0.0)),
        (np.float64(10.787633794225332), np.float64(20.388207304802688), np.float64(0.0)),
        (np.float64(13.515100151982352), np.float64(19.911499207665297), np.float64(0.0)),
        (np.float64(15.80894271642271), np.float64(18.20698834948443), np.float64(0.0)),
        (np.float64(17.567322457325766), np.float64(17.04614632751086), np.float64(0.0)),
        (np.float64(19.809959442021594), np.float64(16.158128560266572), np.float64(0.0)),
    ]),
    
    ('Drop', (np.float64(18.8), np.float64(15.033747339504576), np.float64(0.0))),
    
    ('GoTo', [
        (np.float64(19.809959442021594), np.float64(16.158128560266572), np.float64(0.0)),
        (np.float64(17.567322457325766), np.float64(17.04614632751086), np.float64(0.0)),
        (np.float64(15.80894271642271), np.float64(18.20698834948443), np.float64(0.0)),
        (np.float64(13.515100151982352), np.float64(19.911499207665297), np.float64(0.0)),
        (np.float64(10.787633794225332), np.float64(20.388207304802688), np.float64(0.0)),
        (np.float64(10.262550932881421), np.float64(20.472690694992643), np.float64(0.0)),
        (np.float64(7.643284323684682), np.float64(21.389458742512527), np.float64(0.0)),
        (np.float64(5.996654256919445), np.float64(21.25797517848121), np.float64(0.0)),
        (np.float64(3.4933863949017088), np.float64(20.708861195858578), np.float64(0.0)),
        (np.float64(1.7556904054041524), np.float64(19.633694620398124), np.float64(0.0)),
        (np.float64(0.8), np.float64(15.667038358692167), np.float64(1.0)),
        (np.float64(-0.13141479267040168), np.float64(14.636332487901434), np.float64(0.0)),
    ]),
    
    ('Pick', 0),
    
    ('GoTo', [
        (np.float64(-0.13141479267040168), np.float64(14.636332487901434), np.float64(0.0)),
        # (np.float64(0.8), np.float64(15.667038358692167), np.float64(1.0)),
        (np.float64(1.7556904054041524), np.float64(19.633694620398124), np.float64(0.0)),
        (np.float64(3.4933863949017088), np.float64(20.708861195858578), np.float64(0.0)),
        (np.float64(5.996654256919445), np.float64(21.25797517848121), np.float64(0.0)),
        (np.float64(7.643284323684682), np.float64(21.389458742512527), np.float64(0.0)),
        (np.float64(10.262550932881421), np.float64(20.472690694992643), np.float64(0.0)),
        (np.float64(10.787633794225332), np.float64(20.388207304802688), np.float64(0.0)),
        (np.float64(13.515100151982352), np.float64(19.911499207665297), np.float64(0.0)),
        (np.float64(15.80894271642271), np.float64(18.20698834948443), np.float64(0.0)),
        (np.float64(18.8), np.float64(15.033747339504576), np.float64(1.0)),
        (np.float64(22.34931171265168), np.float64(15.394576545193082), np.float64(2.0)),
    ]),
    
    ('Drop', (np.float64(22.093203435574466), np.float64(16.8), np.float64(2.0))),
    
    ('GoTo', [
        (np.float64(22.34931171265168), np.float64(15.394576545193082), np.float64(2.0)),
        (np.float64(22.093203435574466), np.float64(16.8), np.float64(3.0)),
        (np.float64(22.601130191605726), np.float64(18.690906073809735), np.float64(4.0)),
        (np.float64(23.770730878440425), np.float64(21.297387178028487), np.float64(4.0)),
        (np.float64(22.23785827525131), np.float64(23.757876600232592), np.float64(4.0)),
    ]),
    
    ('Pick', 2),
    
    ('GoTo', [
        (np.float64(22.23785827525131), np.float64(23.757876600232592), np.float64(4.0)),
        (np.float64(23.770730878440425), np.float64(21.297387178028487), np.float64(4.0)),
        (np.float64(22.601130191605726), np.float64(18.690906073809735), np.float64(4.0)),
        (np.float64(22.093203435574466), np.float64(16.8), np.float64(3.0)),
        (np.float64(21.0524090659988), np.float64(14.987410914129672), np.float64(2.0)),
        (np.float64(18.8), np.float64(15.033747339504576), np.float64(1.0)),
        (np.float64(14.622701970030644), np.float64(15.527738049164805), np.float64(0.0)),
        (np.float64(12.651418487071993), np.float64(16.84129304057658), np.float64(0.0)),
    ]),
    
    ('Drop', (np.float64(13.2), np.float64(16.23092211187797), np.float64(0.0))),
    
    ('GoTo', [
        (np.float64(12.651418487071993), np.float64(16.84129304057658), np.float64(0.0)),
        (np.float64(14.622701970030644), np.float64(16.927738049164805), np.float64(0.0)),
        (np.float64(17.081150891255067), np.float64(14.761119683936455), np.float64(0.0)),
        (np.float64(17.751066914068417), np.float64(15.080536653483023), np.float64(0.0)),
    ]),
    
    ('Pick', 1),
    
    ('GoTo', [
        (np.float64(17.751066914068417), np.float64(15.080536653483023), np.float64(0.0)),
        (np.float64(17.567322457325766), np.float64(17.04614632751086), np.float64(0.0)),
        (np.float64(15.80894271642271), np.float64(18.20698834948443), np.float64(0.0)),
        (np.float64(13.515100151982352), np.float64(19.911499207665297), np.float64(0.0)),
        (np.float64(10.787633794225332), np.float64(20.388207304802688), np.float64(0.0)),
        (np.float64(10.262550932881421), np.float64(20.472690694992643), np.float64(0.0)),
        (np.float64(7.643284323684682), np.float64(21.389458742512527), np.float64(0.0)),
        (np.float64(5.996654256919445), np.float64(21.25797517848121), np.float64(0.0)),
        (np.float64(3.4933863949017088), np.float64(20.708861195858578), np.float64(0.0)),
        (np.float64(1.7556904054041524), np.float64(19.633694620398124), np.float64(0.0)),
        (np.float64(0.5955055747967783), np.float64(18.907960092384638), np.float64(0.0)),
        (np.float64(-0.6178826528017431), np.float64(16.203894567747735), np.float64(0.0)),
        (np.float64(-0.13141479267040168), np.float64(14.636332487901434), np.float64(0.0)),
        (np.float64(1.5221295670116604), np.float64(13.069239070703043), np.float64(0.0)),
    ]),
    
    ('Drop', (np.float64(0.8), np.float64(13.225798737001313), np.float64(0.0))),
    
    ('GoTo', [
        (np.float64(1.5221295670116604), np.float64(13.069239070703043), np.float64(0.0)),
        (np.float64(-0.13141479267040168), np.float64(14.636332487901434), np.float64(0.0)),
        (np.float64(-0.6178826528017431), np.float64(16.203894567747735), np.float64(0.0)),
        (np.float64(0.5955055747967783), np.float64(18.907960092384638), np.float64(0.0)),
        (np.float64(1.7556904054041524), np.float64(19.633694620398124), np.float64(0.0)),
        (np.float64(3.4933863949017088), np.float64(20.708861195858578), np.float64(0.0)),
        (np.float64(5.996654256919445), np.float64(21.25797517848121), np.float64(0.0)),
        (np.float64(7.643284323684682), np.float64(21.389458742512527), np.float64(0.0)),
        (np.float64(10.262550932881421), np.float64(20.472690694992643), np.float64(0.0)),
        (np.float64(10.787633794225332), np.float64(20.388207304802688), np.float64(0.0)),
        (np.float64(13.515100151982352), np.float64(19.911499207665297), np.float64(0.0)),
        (np.float64(15.44201566849232), np.float64(18.366079952262684), np.float64(0.0)),
        (np.float64(14.622701970030644), np.float64(15.527738049164805), np.float64(0.0)),
        (np.float64(12.651418487071993), np.float64(16.84129304057658), np.float64(0.0)),
    ]),
    
    ('Pick', 2),
    
    ('GoTo', [
        (np.float64(12.651418487071993), np.float64(16.84129304057658), np.float64(0.0)),
        (np.float64(14.622701970030644), np.float64(15.527738049164805), np.float64(0.0)),
        (np.float64(15.44201566849232), np.float64(18.366079952262684), np.float64(0.0)),
        (np.float64(13.515100151982352), np.float64(19.911499207665297), np.float64(0.0)),
        (np.float64(10.787633794225332), np.float64(20.388207304802688), np.float64(0.0)),
        (np.float64(10.262550932881421), np.float64(20.472690694992643), np.float64(0.0)),
        (np.float64(7.643284323684682), np.float64(21.389458742512527), np.float64(0.0)),
        (np.float64(5.996654256919445), np.float64(21.25797517848121), np.float64(0.0)),
        (np.float64(3.4933863949017088), np.float64(20.708861195858578), np.float64(0.0)),
        (np.float64(1.7556904054041524), np.float64(19.633694620398124), np.float64(0.0)),
        (np.float64(0.5955055747967783), np.float64(18.907960092384638), np.float64(0.0)),
        (np.float64(-0.6178826528017431), np.float64(16.203894567747735), np.float64(0.0)),
        (np.float64(0.8), np.float64(13.225798737001313), np.float64(1.0)),
        (np.float64(4.850576287128186), np.float64(12.900083517227444), np.float64(2.0)),
        (np.float64(5.710733245203751), np.float64(13.699904126962089), np.float64(2.0)),
        (np.float64(6.841551040462982), np.float64(15.148362751419107), np.float64(2.0)),
        (np.float64(7.15532273259052), np.float64(16.682256502131413), np.float64(2.0)),
    ]),
    
    ('Drop', (np.float64(7.2), np.float64(15.740782553633101), np.float64(2.0))),
    
    ('GoTo', [
        (np.float64(7.15532273259052), np.float64(16.982256502131413), np.float64(2.0)),
        (np.float64(8.15532273259052), np.float64(17.982256502131413), np.float64(2.0)),
        (np.float64(10.01867281796113), np.float64(15.892702894413931), np.float64(3.0)),
        (np.float64(7.2), np.float64(15.740782553633101), np.float64(4.0)),
        (np.float64(3.579457091470507), np.float64(16.85256082854877), np.float64(5.0)),
    ])]

    """ H=2 """
    # 1) define where to show the reach sphere (global waypoint indices)
    sphere_wp_idxs = {36, 40, 42, 43}   # whatever indices best illustrate “can’t reach without the box”
    reach_XY = 3
    reach_Z = 1.5           # your hard-coded max single-step distance
    
    # 2) define all possible plane-connectivity lines (once)
    all_lines = [
        ((5,12,2), (8,16,3)),       # T1-O
        ((5,12,0), (5,12,2)),       # G-T1
        ((5,7,1), (5,12,2)),      # B1-T1
        ((4,4,0), (5,7,1)),       # G-T1
        ((8,16,3),(14,19,3)),       # O-B2
        ((14,19,3),(22,17,3)),      # B2-T2
        ((8,16,3),(22,17,3)),       # O-T2
    ]
    
    # 3) schedule which subset to show as connectivity evolves
    lines_at = [
        [0],         # start: nothing connected
        [0,1],
        [0,1,6],    
        ]
    
    # 4) when to advance to each subset (by waypoint index)
    line_switch_wp_idxs = [0, 13, 29]  # must be non-decreasing; len == len(lines_at)
    
    """ H= 6 """
        
    # 1) define where to show the reach sphere (global waypoint indices)
    sphere_wp_idxs = {16,17,18, 21,22, 40,41,45,46,50,51,52,53,100,101,105,106,107}   # whatever indices best illustrate “can’t reach without the box”
    reach_XY = 3
    reach_Z = 1.5             # your hard-coded max single-step distance
    
    # 2) define all possible plane-connectivity lines (once)
    all_lines = [
        # table1 - O
        ((7.0, 14.0, 2.0), (10.5, 16.0, 3.0)),
        # ground - table1
        ((7.0, 14.0, 0.0), (7.0, 14.0, 2.0)),
        # O - Y
        ((10.5, 16.0, 3.0), (4.0, 16.0, 5.0)),
        # ground - table2
        ((24.5, 17.5, 0.0), (24.5, 17.5, 2.0)),
        # table2 - R
        ((24.5, 17.5, 2.0), (22.5, 21.0, 4.0)),
    ]
    
    # 3) schedule which subset to show as connectivity evolves
    lines_at = [
        [0],    
        [0,1],
        [0,1,3],    
        [0,3],
        [0,3,4],
        [0,4],
        [0,1,4],
        [0,1,2,4],    
        ]
    
    # 4) when to advance to each subset (by waypoint index)
    line_switch_wp_idxs = [0, 15, 27, 34, 44, 80, 94, 103]  # must be non-decreasing; len == len(lines_at)
    
    animate_plan(plan_actions, env_planes, block_info_list)