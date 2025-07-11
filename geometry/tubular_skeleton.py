import numpy as np
import pyvista as pv
from scipy.interpolate import CubicHermiteSpline
from collections import defaultdict
from typing import Dict, List, Tuple, Set, Optional

__all__ = [
    "SkeletonModel",
    "catmull_rom_spline_polydata",
]

# -----------------------------------------------------------------------------
#   Low‑level utilities
# -----------------------------------------------------------------------------

_TOL = 1e-6  # distance tolerance used to match shared junction points


def _normalize(v: np.ndarray) -> np.ndarray:
    """Return *v* normalised (safe if ‖v‖ ≈ 0)."""
    n = np.linalg.norm(v)
    return v / n if n > 0 else v


# -----------------------------------------------------------------------------
#   Core Catmull–Rom helper (same as before, but extracted for reuse)
# -----------------------------------------------------------------------------

def catmull_rom_spline_polydata(
    points: np.ndarray,
    samples_per_segment: int = 25,
    closed: bool = False,
    start_tangent: Optional[np.ndarray] = None,
    end_tangent: Optional[np.ndarray] = None,
) -> pv.PolyData:
    """Return a centripetal Catmull–Rom curve through *points* as PolyData."""

    if points.shape[0] < 4:
        raise ValueError("Catmull–Rom requires at least 4 control points.")

    # --- centripetal knot vector (α = 0.5) ----------------------------------
    alpha = 0.5
    t = np.zeros(len(points))
    for i in range(1, len(t)):
        t[i] = t[i - 1] + np.linalg.norm(points[i] - points[i - 1]) ** alpha

    # --- default derivatives -------------------------------------------------
    m = np.empty_like(points)
    m[1:-1] = (points[2:] - points[:-2]) / (t[2:, None] - t[:-2, None])

    if closed:
        m[0] = (points[1] - points[-2]) / (t[1] - (t[-2] - t[-1]))
        m[-1] = m[0]
    else:
        m[0] = (points[1] - points[0]) / (t[1] - t[0])
        m[-1] = (points[-1] - points[-2]) / (t[-1] - t[-2])

    # --- optional end‑point clamping ----------------------------------------
    if (not closed) and start_tangent is not None:
        m[0] = _normalize(start_tangent) * np.linalg.norm(m[0])
    if (not closed) and end_tangent is not None:
        m[-1] = _normalize(end_tangent) * np.linalg.norm(m[-1])

    # --- Hermite splines -----------------------------------------------------
    xs = CubicHermiteSpline(t, points[:, 0], m[:, 0])
    ys = CubicHermiteSpline(t, points[:, 1], m[:, 1])
    zs = CubicHermiteSpline(t, points[:, 2], m[:, 2])

    n_seg = len(points) - 1 if not closed else len(points)
    n_eval = n_seg * samples_per_segment + 1
    t_eval = np.linspace(t[0], t[-1], n_eval)

    curve_xyz = np.column_stack([xs(t_eval), ys(t_eval), zs(t_eval)])

    # --- wrap into PolyData --------------------------------------------------
    n_pts = curve_xyz.shape[0]
    poly_line = np.hstack(([n_pts], np.arange(n_pts))).astype(np.int64)
    pd = pv.PolyData(curve_xyz)
    pd.lines = poly_line
    return pd


# -----------------------------------------------------------------------------
#   Skeleton model – dynamic & efficient recomputation
# -----------------------------------------------------------------------------

class SkeletonModel:
    """Template Circle‑of‑Willis skeleton with *mutable* junction knots.

    Parameters
    ----------
    inlet : dict[str, list[tuple[float, float, float]]]
        Mapping *name → control‑point list* for inlet arteries – these *define*
        tangent directions for connected branches.
    outlet, communicating : same structure for the remaining artery groups.
    samples_per_segment : int, optional
        Evaluation density when sampling each spline.
    tol : float, optional
        Distance tolerance used to identify shared junctions between arteries.
    """

    def __init__(
        self,
        inlet: Dict[str, List[Tuple[float, float, float]]],
        outlet: Dict[str, List[Tuple[float, float, float]]],
        communicating: Dict[str, List[Tuple[float, float, float]]],
        samples_per_segment: int = 25,
        tol: float = _TOL,
    ) -> None:
        self.tol = tol
        self.samples_per_segment = samples_per_segment

        # Store control‑points per artery as (N,3) float arrays
        self._points: Dict[str, np.ndarray] = {
            **{n: np.asarray(p, float) for n, p in inlet.items()},
            **{n: np.asarray(p, float) for n, p in outlet.items()},
            **{n: np.asarray(p, float) for n, p in communicating.items()},
        }
        self._inlet_names: Set[str] = set(inlet)

        # caches
        self._junction_map: Dict[int, List[Tuple[str, int]]] = {}
        self._splines: Dict[str, pv.PolyData] = {}

        self._rebuild_junction_index()
        self._recompute_all()

    # ------------------------------------------------------------------
    #   External interface
    # ------------------------------------------------------------------

    def move_knot(
        self,
        artery: str,
        index: int,
        new_xyz: Tuple[float, float, float],
    ) -> Set[str]:
        """Move a specific control point *in‑place* and recompute dependencies.

        Returns
        -------
        set[str]
            Names of arteries whose splines were recomputed.
        """
        if artery not in self._points:
            raise KeyError(f"Unknown artery '{artery}'.")
        pts = self._points[artery]
        if not (0 <= index < len(pts)):
            raise IndexError("knot index out of range")

        # 1. remember old coordinate, write the new one
        old_coord = pts[index].copy()
        pts[index] = new_xyz

        # 2. rebuild the junction index (cheap – only endpoints are inspected)
        self._rebuild_junction_index()

        # 3. determine arteries that shared *either* the old or new coord
        affected: Set[str] = {artery}
        affected |= self._arteries_sharing_point(old_coord)
        affected |= self._arteries_sharing_point(new_xyz)

        # 4. recompute affected splines
        for art in affected:
            self._splines[art] = self._compute_spline(art)
        return affected

    # Convenience: accessors --------------------------------------------------

    def get_spline(self, artery: str) -> pv.PolyData:
        return self._splines[artery]

    def all_splines(self) -> Dict[str, pv.PolyData]:
        return self._splines

    # ------------------------------------------------------------------
    #   Internal helpers
    # ------------------------------------------------------------------

    def _recompute_all(self) -> None:
        """Recompute *every* spline (used only at init or full reset)."""
        for art in self._points:
            self._splines[art] = self._compute_spline(art)

    # ---- junction‑index construction ---------------------------------

    def _rebuild_junction_index(self) -> None:
        """Index endpoints that coincide (within *tol*) across arteries.

        We consider only **first** and **last** control‑point of each artery
        because only these contribute to tangent sharing.
        """
        self._junction_map = {}
        nodes: List[np.ndarray] = []  # representative coordinate per node

        def _find_node_id(p: np.ndarray) -> int:
            for nid, rep in enumerate(nodes):
                if np.linalg.norm(rep - p) < self.tol:
                    return nid
            return -1

        for name, pts in self._points.items():
            for idx in (0, -1):
                p = pts[idx]
                nid = _find_node_id(p)
                if nid == -1:  # create new node
                    nid = len(nodes)
                    nodes.append(p.copy())
                self._junction_map.setdefault(nid, []).append((name, idx))

    def _arteries_sharing_point(self, xyz: np.ndarray) -> Set[str]:
        """Return all arteries that have a junction ≈ *xyz*."""
        for members in self._junction_map.values():
            rep = self._points[members[0][0]][members[0][1]]  # representative
            if np.linalg.norm(rep - xyz) < self.tol:
                return {art for art, _ in members}
        return set()

    # ---- spline construction ------------------------------------------

    def _parent_tangent(self, parent_pts: np.ndarray, idx: int) -> np.ndarray:
        """Simple 3‑point finite‑difference tangent."""
        if idx == 0:
            return parent_pts[1] - parent_pts[0]
        elif idx == len(parent_pts) - 1:
            return parent_pts[-1] - parent_pts[-2]
        else:
            return parent_pts[idx + 1] - parent_pts[idx - 1]

    def _matching_tangent(self, child: str, point: np.ndarray) -> Optional[np.ndarray]:
        """Return tangent from *another* artery sharing *point*, if any."""
        for members in self._junction_map.values():
            shared = [m for m in members if np.linalg.norm(self._points[m[0]][m[1]] - point) < self.tol]
            if shared:
                for art, idx in members:
                    if art != child and art in self._inlet_names:  # only inlet provide tangents
                        return self._parent_tangent(self._points[art], idx)
        return None

    def _compute_spline(self, name: str) -> pv.PolyData:
        pts = self._points[name]
        start_tan = self._matching_tangent(name, pts[0])
        end_tan = self._matching_tangent(name, pts[-1])
        return catmull_rom_spline_polydata(
            pts.copy(),
            samples_per_segment=self.samples_per_segment,
            start_tangent=start_tan,
            end_tangent=end_tan,
        )


# -----------------------------------------------------------------------------
#   Example usage (run as a script to test)
# -----------------------------------------------------------------------------

if __name__ == "__main__":
    # --- control‑point dictionaries identical to the previous example --------
    inlet_arteries = {
        "BA": [(-4.5, 0, -5), (-4, 0, -4), (-4, 0, -3), (-4, 0, -2), (-4.5, 0, -1), (-5, 0, 0)],
        "RICA": [(2, -4, -4.5), (4, -4, -4), (5.5, -4, -3), (5, -4, -1.5), (3.5, -4, -1),
                  (2, -4, 0), (2, -3.5, 1.5), (3, -3, 2.5)],
        "LICA": [(2, 4, -4.5), (4, 4, -4), (5.5, 4, -3), (5, 4, -1.5), (3.5, 4, -1),
                  (2, 4, 0), (2, 3.5, 1.5), (3, 3, 2.5)],
    }

    outlet_arteries = {
        "RPCA": [(-5, 0, 0), (-5, -1, 0.5), (-5, -2, 0.5), (-5, -3, 0), (-5, -4, 0),
                  (-5, -5, 0), (-5, -6, -0.5), (-5.5, -7, -0.5), (-6, -8, 0), (-6, -9, 1)],
        "LPCA": [(-5, 0, 0), (-5, 1, 0.5), (-5, 2, 0.5), (-5, 3, 0), (-5, 4, 0),
                  (-5, 5, 0), (-5, 6, -0.5), (-5.5, 7, -0.5), (-6, 8, 0), (-6, 9, 1)],
        "RMCA": [(3, -3, 2.5), (3.5, -4, 3.5), (4, -5, 4), (4.5, -6, 4), (4.5, -7, 4),
                  (4.5, -8, 4), (4.5, -9, 4.5)],
        "LMCA": [(3, 3, 2.5), (3.5, 4, 3.5), (4, 5, 4), (4.5, 6, 4), (4.5, 7, 4),
                  (4.5, 8, 4), (4.5, 9, 4.5)],
        "RACA": [(3, -3, 2.5), (4, -2, 3.5), (4.5, -1, 4), (5.5, -0.5, 5), (6, -0.5, 6.5),
                  (6.5, -0.5, 7.5), (7, -0.5, 8.5), (7, -1, 9.5)],
        "LACA": [(3, 3, 2.5), (4, 2, 3.5), (4.5, 1, 4), (5.5, 0.5, 5), (6, 0.5, 6.5),
                  (6.5, 0.5, 7.5), (7, 0.5, 8.5), (7, 1, 9.5)],
    }

    communicating_arteries = {
        "RPCOM": [(-5, -4, 0), (-4, -3.5, 0), (-3, -3, -0.5), (-2, -2.5, -0.5),
                   (-1, -2.5, -0.5), (0, -3, 0), (1, -3.5, 0.5), (2, -4, 0)],
        "LPCOM": [(-5, 4, 0), (-4, 3.5, 0), (-3, 3, -0.5), (-2, 2.5, -0.5),
                   (-1, 2.5, -0.5), (0, 3, 0), (1, 3.5, 0.5), (2, 4, 0)],
        "ACOM": [(4.5, -1, 4), (4.5, -0.5, 4), (4.5, 0, 4), (4.5, 0.5, 4), (4.5, 1, 4)],
    }

    # --- build skeleton ------------------------------------------------------
    skel = SkeletonModel(inlet_arteries, outlet_arteries, communicating_arteries)

    # Move one control point to demonstrate dynamic update
    moved = skel.move_knot("RPCA", 0, (-5.2, -0.2, 0.1))
    print("Recomputed:", moved)

    # Visualise ---------------------------------------------------------------
    plotter = pv.Plotter()
    for poly in skel.all_splines().values():
        plotter.add_mesh(poly, line_width=6, render_lines_as_tubes=True)
    plotter.show()
