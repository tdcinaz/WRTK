"""template_skeleton.py
================================================
Stage‑1 scaffold for building a *template* Circle‑of‑Willis centre‑line skeleton.

This **fixed** edition separates *graph topology* from the cached spline
samples, so edges are now always generated when you ask to plot or export the
skeleton.
"""
from __future__ import annotations

import itertools
from dataclasses import dataclass, field
from typing import Dict, Iterable, List, Set, Tuple

import numpy as np
from scipy.interpolate import CubicHermiteSpline

# -----------------------------------------------------------------------------
# Data containers
# -----------------------------------------------------------------------------

@dataclass
class Node:
    id: str
    xyz: np.ndarray  # shape (3,)

    def __post_init__(self) -> None:  # sanity guard
        self.xyz = np.asarray(self.xyz, float)
        if self.xyz.shape != (3,):
            raise ValueError("xyz must be length‑3")

EdgeKey = Tuple[str, str]

def ek(a: str, b: str) -> EdgeKey:  # canonical key (unordered)
    return tuple(sorted((a, b)))

# -----------------------------------------------------------------------------
# Core skeleton graph
# -----------------------------------------------------------------------------

@dataclass
class TemplateSkeleton:
    """Graph of nodes + cached C¹‑continuous splines on each edge."""

    samples_per_edge: int = 100

    _nodes: Dict[str, Node] = field(default_factory=dict, init=False, repr=False)
    _adjacent: Set[EdgeKey] = field(default_factory=set, init=False, repr=False)  # topology only
    _edges: Dict[EdgeKey, np.ndarray] = field(default_factory=dict, init=False, repr=False)  # cache

    # ---------------------------------------------------------------------
    # Construction
    # ---------------------------------------------------------------------

    def add_node(self, node_id: str, xyz: Iterable[float]) -> None:
        """Insert or update a control point."""
        self._nodes[node_id] = Node(str(node_id), xyz)
        # Moving or adding → purge curves touching this node
        self._invalidate_edges(node_id)

    def connect(self, a: str, b: str) -> None:
        """Declare an undirected edge between *a* and *b*."""
        if a not in self._nodes or b not in self._nodes:
            raise KeyError("Both nodes must exist before connecting")
        self._adjacent.add(ek(a, b))
        self._edges.pop(ek(a, b), None)  # ensure spline is regenerated

    # ---------------------------------------------------------------------
    # Modification
    # ---------------------------------------------------------------------

    def move_node(self, node_id: str, new_xyz: Iterable[float]) -> None:
        if node_id not in self._nodes:
            raise KeyError(node_id)
        self._nodes[node_id].xyz = np.asarray(new_xyz, float)
        self._invalidate_edges(node_id)

    def _invalidate_edges(self, node_id: str) -> None:
        for e in list(self._edges):
            if node_id in e:
                self._edges.pop(e)

    # ---------------------------------------------------------------------
    # Queries
    # ---------------------------------------------------------------------

    def node_coords(self, node_id: str) -> np.ndarray:
        return self._nodes[node_id].xyz

    def edge_curve(self, a: str, b: str) -> np.ndarray:
        key = ek(a, b)
        if key not in self._adjacent:
            raise KeyError(f"Nodes {a}–{b} are not connected")
        if key not in self._edges:  # lazy generation
            self._edges[key] = self._build_curve(a, b)
        return self._edges[key]

    # ------------------------------------------------------------------
    # Export & visualisation helpers
    # ------------------------------------------------------------------

    def as_line_set(self) -> Tuple[np.ndarray, np.ndarray]:
        """Return (points, lines) arrays for *pyvista.PolyData*."""
        pts_list: List[np.ndarray] = []
        lines: List[int] = []
        offset = 0
        for a, b in self._adjacent:
            curve = self.edge_curve(a, b)  # ensures cached
            n = len(curve)
            pts_list.append(curve)
            lines.extend([n] + list(range(offset, offset + n)))
            offset += n
        if not pts_list:
            return np.empty((0, 3)), np.empty((0,), int)
        return np.vstack(pts_list), np.asarray(lines, int)

    def to_pyvista(self):  # noqa: D401
        if not _has_pyvista():
            raise RuntimeError("PyVista not installed; `pip install pyvista`.")
        import pyvista as pv  # local import
        pts, lines = self.as_line_set()
        poly = pv.PolyData()
        if pts.size:
            poly.points = pts
            poly.lines = lines
        return poly

    # ------------------------------------------------------------------
    # Internal – spline generation
    # ------------------------------------------------------------------

    def _build_curve(self, a: str, b: str) -> np.ndarray:
        p0, p1 = self._nodes[a].xyz, self._nodes[b].xyz
        t0 = self._approx_tangent(node=a, other=b)
        t1 = self._approx_tangent(node=b, other=a)

        ts = np.linspace(0.0, 1.0, self.samples_per_edge)
        curve = np.empty((self.samples_per_edge, 3))
        for d in range(3):
            cs = CubicHermiteSpline([0, 1], [p0[d], p1[d]], [t0[d], t1[d]])
            curve[:, d] = cs(ts)
        return curve

    def _approx_tangent(self, *, node: str, other: str) -> np.ndarray:
        neighbours = [n for e in self._adjacent for n in e if node in e and n != node]
        neighbours = [n for n in neighbours if n != other]
        if neighbours:
            prev = neighbours[0]
            tangent = self._nodes[other].xyz - self._nodes[prev].xyz
        else:  # endpoint
            tangent = self._nodes[other].xyz - self._nodes[node].xyz
        norm = np.linalg.norm(tangent)
        return tangent / (norm + 1e-8)

# -----------------------------------------------------------------------------
# Convenience
# -----------------------------------------------------------------------------

def _has_pyvista() -> bool:  # pragma: no cover
    try:
        import pyvista  # noqa: F401
    except ModuleNotFoundError:
        return False
    return True

# -----------------------------------------------------------------------------
# Demo (run `python template_skeleton.py`)
# -----------------------------------------------------------------------------

if __name__ == "__main__":
    sk = TemplateSkeleton(samples_per_edge=50)

    # Nodes (toy example)
    sk.add_node("BA", (0, 0, -10))
    sk.add_node("PCA_R", (5, 0, -5))
    sk.add_node("PCA_L", (-5, 0, -5))
    sk.add_node("ICA_R", (7, 0, 0))
    sk.add_node("ICA_L", (-7, 0, 0))
    sk.add_node("ACA_R", (3, 0, 5))
    sk.add_node("ACA_L", (-3, 0, 5))

    # Edges
    for a, b in [
        ("BA", "PCA_R"), ("BA", "PCA_L"),
        ("PCA_R", "ICA_R"), ("PCA_L", "ICA_L"),
        ("ICA_R", "ACA_R"), ("ICA_L", "ACA_L"),
        ("ACA_R", "ACA_L"), ("ICA_R", "ICA_L"),
    ]:
        sk.connect(a, b)

    sk.to_pyvista().plot(line_width=4)
    # Move a node – curves auto‑invalidate
    sk.move_node("ICA_R", (10, 0, 0))

    sk.to_pyvista().plot(line_width=4)

