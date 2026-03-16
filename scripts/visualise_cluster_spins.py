#!/usr/bin/env python3
"""
Spin + bond visualizer.

Spins file columns (tab-separated):
    &s  X[0]  X[1]  X[2]  deleted  cluster_root

Bonds file columns (tab-separated):
    &site_a  &site_b

Rules:
  - Normal spins (deleted=0, cluster_root=NULL)  → black filled circle
  - Deleted spins (deleted=1)                    → hollow red circle
  - Cluster spins (cluster_root != NULL)          → filled square, color unique per cluster
  - Bonds between non-cluster spins               → dim grey lines
  - Bonds where both endpoints in same cluster    → colored lines (cluster color)
  - Bonds crossing cluster boundaries / mixed     → white lines

Uses FURY for GPU-accelerated rendering, efficient for 10k+ spins / 100k+ bonds.

Usage:
    python visualise_spins.py spins.csv [bonds.csv]
"""

import argparse
import sys
import numpy as np


# ---------------------------------------------------------------------------
# Parsing
# ---------------------------------------------------------------------------

NULL_PTRS = {'0', '0x0', '(nil)', '0x0000000000000000'}


def parse_spins(path: str):
    """
    Parse spins CSV.  Returns:
        ptr_to_idx  : dict[str, int]   pointer string -> row index
        positions   : (N,3) float64
        deleted     : (N,)  int32
        cluster_ids : (N,)  int32      0 = no cluster, 1..K = cluster index
    """
    ptr_to_idx = {}
    positions, deleted, cluster_ids = [], [], []
    cluster_map: dict[str, int] = {}
    next_cid = [1]

    with open(path) as fh:
        for lineno, line in enumerate(fh, 1):
            line = line.strip()
            if not line or line.startswith('#'):
                continue
            parts = line.split('\t')
            if len(parts) != 6:
                print(f"[spins] Warning: skipping malformed line {lineno} "
                      f"(expected 6 cols, got {len(parts)}): {line!r}", file=sys.stderr)
                continue
            try:
                site_ptr = parts[0].strip()
                x, y, z  = float(parts[1]), float(parts[2]), float(parts[3])
                del_flag = int(parts[4])
                root_ptr = parts[5].strip()
            except ValueError as e:
                print(f"[spins] Warning: parse error at line {lineno}: {e}", file=sys.stderr)
                continue

            idx = len(positions)
            ptr_to_idx[site_ptr] = idx
            positions.append((x, y, z))
            deleted.append(del_flag)

            if root_ptr in NULL_PTRS:
                cluster_ids.append(0)
            else:
                if root_ptr not in cluster_map:
                    cluster_map[root_ptr] = next_cid[0]
                    next_cid[0] += 1
                cluster_ids.append(cluster_map[root_ptr])

    return (ptr_to_idx,
            np.array(positions,    dtype=np.float64),
            np.array(deleted,      dtype=np.int32),
            np.array(cluster_ids,  dtype=np.int32))


def parse_bonds(path: str, ptr_to_idx: dict) -> np.ndarray:
    """
    Parse bonds CSV.  Returns (M, 2) int32 array of spin indices.
    Bonds referencing unknown pointers are skipped with a warning.
    """
    bonds = []
    skipped = 0
    with open(path) as fh:
        for lineno, line in enumerate(fh, 1):
            line = line.strip()
            if not line or line.startswith('#'):
                continue
            parts = line.split('\t')
            if len(parts) != 2:
                print(f"[bonds] Warning: skipping malformed line {lineno}: {line!r}", file=sys.stderr)
                continue
            pa, pb = parts[0].strip(), parts[1].strip()
            ia, ib = ptr_to_idx.get(pa), ptr_to_idx.get(pb)
            if ia is None or ib is None:
                skipped += 1
                continue
            bonds.append((ia, ib))

    if skipped:
        print(f"[bonds] Warning: {skipped} bonds skipped (unknown pointers)", file=sys.stderr)

    return np.array(bonds, dtype=np.int32) if bonds else np.empty((0, 2), dtype=np.int32)


# ---------------------------------------------------------------------------
# Color helpers
# ---------------------------------------------------------------------------

def cluster_colormap(n_clusters: int) -> np.ndarray:
    """Return (n_clusters+1, 3) float32 RGB; row 0 is black (unused)."""
    colors = np.zeros((n_clusters + 1, 3), dtype=np.float32)
    if n_clusters == 0:
        return colors
    golden = 0.618033988749895
    hues = np.mod(np.arange(n_clusters) * golden, 1.0)
    for i, h in enumerate(hues):
        hi = int(h * 6)
        f  = h * 6 - hi
        q, t = 1.0 - f, f
        sectors = [
            (1.0, t,   0.0),
            (q,   1.0, 0.0),
            (0.0, 1.0, t  ),
            (0.0, q,   1.0),
            (t,   0.0, 1.0),
            (1.0, 0.0, q  ),
        ]
        colors[i + 1] = sectors[hi % 6]
    return colors


# ---------------------------------------------------------------------------
# Visualisation
# ---------------------------------------------------------------------------

def visualise(positions: np.ndarray, deleted: np.ndarray,
              cluster_ids: np.ndarray, bonds: np.ndarray):
    try:
        from fury import actor, window
    except ImportError:
        sys.exit("FURY is not installed.  Run:  pip install fury")

    n = len(positions)
    print(f"Loaded {n} spins, {len(bonds)} bonds.")

    pos    = positions.astype(np.float32)
    centre = pos.mean(axis=0)
    span   = float(np.ptp(pos, axis=0).max())
    scale  = max(span / 200.0, 0.5)

    # ---- spin masks --------------------------------------------------------
    mask_cluster = cluster_ids != 0
    mask_deleted = (deleted == 1) & ~mask_cluster
    mask_normal  = ~mask_cluster & ~mask_deleted

    n_clusters = int(cluster_ids.max()) if mask_cluster.any() else 0
    cmap = cluster_colormap(n_clusters)

    print(f"  Normal:  {mask_normal.sum()}")
    print(f"  Deleted: {mask_deleted.sum()}")
    print(f"  Cluster: {mask_cluster.sum()} across {n_clusters} cluster(s)")
    print(f"  Bonds:   {len(bonds)}")

    scene = window.Scene()
    scene.SetBackground(0.12, 0.12, 0.14)

    # ===== BONDS ============================================================
    if len(bonds) > 0:
        cid_a = cluster_ids[bonds[:, 0]]
        cid_b = cluster_ids[bonds[:, 1]]

        # Same non-zero cluster on both ends
        mask_same  = (cid_a != 0) & (cid_b != 0) & (cid_a == cid_b)
        # At least one end in a cluster, but not the same
        mask_cross = ((cid_a != 0) | (cid_b != 0)) & ~mask_same
        # Both ends outside any cluster
        mask_plain = ~mask_same & ~mask_cross

        def add_bonds_uniform(bmask, rgb, linewidth, opacity):
            """All bonds in bmask share a single uniform color (tuple/list of 3 floats)."""
            sel  = bonds[bmask]
            pts_a = pos[sel[:, 0]]
            pts_b = pos[sel[:, 1]]
            # actor.line wants a list of (N,3) polylines; each 2-point segment is one line
            segs = [np.array([a, b], dtype=np.float32) for a, b in zip(pts_a, pts_b)]
            act  = actor.line(segs, colors=rgb, linewidth=linewidth, opacity=opacity)
            scene.add(act)

        def add_bonds_colored(bmask, color_per_seg, linewidth, opacity):
            """Each bond in bmask gets its own color; color_per_seg is (M,3) float32."""
            sel   = bonds[bmask]
            pts_a = pos[sel[:, 0]]
            pts_b = pos[sel[:, 1]]
            colors = color_per_seg
            segs   = [np.array([a, b], dtype=np.float32) for a, b in zip(pts_a, pts_b)]
            # Pass one RGB tuple per line (not per vertex)
            cols   = [tuple(c) for c in colors]
            act    = actor.line(segs, colors=cols, linewidth=linewidth, opacity=opacity)
            scene.add(act)

        if mask_plain.any():
            add_bonds_uniform(mask_plain, (0.45, 0.45, 0.45), linewidth=0.8, opacity=0.45)

        if mask_same.any():
            seg_rgb = cmap[cid_a[mask_same]].astype(np.float32)
            add_bonds_colored(mask_same, seg_rgb, linewidth=1.6, opacity=0.85)

        if mask_cross.any():
            add_bonds_uniform(mask_cross, (1.0, 1.0, 1.0), linewidth=1.2, opacity=0.7)


    # ===== SPINS ============================================================

    # -- 1. Normal spins: near-black filled circles -------------------------
    if mask_normal.any():
        npos = pos[mask_normal]
        ncol = np.tile([0.05, 0.05, 0.05, 1.0], (len(npos), 1)).astype(np.float32)
        scene.add(actor.markers(npos, colors=ncol, scales=scale,
                                marker='o', marker_opacity=1.0, edge_width=0.0))

    # -- 2. Deleted spins: hollow red circles --------------------------------
    if mask_deleted.any():
        dpos = pos[mask_deleted]
        dcol = np.tile([1.0, 0.0, 0.0, 0.0], (len(dpos), 1)).astype(np.float32)
        scene.add(actor.markers(dpos, colors=dcol, scales=scale * 1.2,
                                marker='o', marker_opacity=0.0,
                                edge_width=0.15,
                                edge_color=(1.0, 0.0, 0.0), edge_opacity=1.0))

    # -- 3. Cluster spins: colored squares -----------------------------------
    if mask_cluster.any():
        cpos    = pos[mask_cluster]
        cids    = cluster_ids[mask_cluster]
        ccols   = cmap[cids]
        ccols_a = np.concatenate([ccols,
                                  np.ones((len(ccols), 1), dtype=np.float32)], axis=1)
        scene.add(actor.markers(cpos, colors=ccols_a, scales=scale * 1.3,
                                marker='s', marker_opacity=0.95,
                                edge_width=0.08,
                                edge_color=(1.0, 1.0, 1.0), edge_opacity=0.6))

    # ===== Camera ===========================================================
    scene.set_camera(
        position=(centre[0], centre[1], centre[2] + span * 2.0),
        focal_point=tuple(centre),
        view_up=(0.0, 1.0, 0.0),
    )

    print("\nControls:  Left-drag = rotate | Middle-drag = pan | Scroll = zoom | Q = quit")
    window.show(scene, title="Spin Lattice Visualiser", size=(1280, 960), reset_camera=False)



# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Visualise spin-lattice data with FURY.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument("spins", nargs='?', help="Spins CSV file")
    parser.add_argument("bonds", nargs='?', help="Bonds CSV file")
#    parser.add_argument("min_cluster_size", type=int, default=None, help="clusters smaller than this are not plotted")
#    parser.add_argument("max_cluster_size", type=int, default=None, help="clusters larger than this are not plotted")
    args = parser.parse_args()

    if args.spins and args.bonds:
        sp, bp = args.spins, args.bonds
    elif args.spins and not args.bonds:
        import os
        bp = args.spins.replace('spins', 'bonds')
        sp = args.spins
        if not os.path.exists(bp):
            parser.error(f"Bonds file not given and guessed path not found: {bp}")
        print(f"Using guessed bonds file: {bp}")
    else:
        parser.print_help()
        sys.exit(1)

    ptr_to_idx, positions, deleted, cluster_ids = parse_spins(sp)
    bonds = parse_bonds(bp, ptr_to_idx)

    if len(positions) == 0:
        sys.exit("No valid spin data found.")

    visualise(positions, deleted, cluster_ids, bonds)


if __name__ == "__main__":
    main()
