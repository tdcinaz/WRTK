import numpy as np
import pyvista as pv
from scipy.interpolate import CubicHermiteSpline
from typing import Dict, List, Tuple, Set, Optional

__all__ = [
    "SkeletonModel",
    "catmull_rom_spline_polydata",
]

# -----------------------------------------------------------------------------
#   Low‑level utilities
# -----------------------------------------------------------------------------

_TOL = 1e-6  # distance tolerance used to match shared junction points


def _normalize(vec: np.ndarray) -> np.ndarray:
    """Return *vec* normalised (safe if ‖vec‖ ≈ 0)."""
    n = np.linalg.norm(vec)
    return vec / n if n > 0 else vec


# -----------------------------------------------------------------------------
#   Core Catmull–Rom helper (centripetal, optional tangent clamping)
# -----------------------------------------------------------------------------

def catmull_rom_spline_polydata(
    points: np.ndarray,
    samples_per_segment: int = 25,
    closed: bool = False,
    start_tangent: Optional[np.ndarray] = None,
    end_tangent: Optional[np.ndarray] = None,
) -> pv.PolyData:
    """Return a centripetal Catmull‑Rom curve through *points* as PolyData."""

    if points.shape[0] < 4:
        raise ValueError("Catmull–Rom requires at least 4 control points.")

    # -- centripetal knot vector (α = 0.5) -----------------------------------
    alpha = 0.5
    t = np.zeros(len(points))
    for i in range(1, len(t)):
        t[i] = t[i - 1] + np.linalg.norm(points[i] - points[i - 1]) ** alpha

    # -- default first‑derivative estimates ----------------------------------
    m = np.empty_like(points)
    m[1:-1] = (points[2:] - points[:-2]) / (t[2:, None] - t[:-2, None])
    if closed:
        m[0] = (points[1] - points[-2]) / (t[1] - (t[-2] - t[-1]))
        m[-1] = m[0]
    else:
        m[0] = (points[1] - points[0]) / (t[1] - t[0])
        m[-1] = (points[-1] - points[-2]) / (t[-1] - t[-2])

    # -- optional tangent clamping -------------------------------------------
    if (not closed) and start_tangent is not None:
        m[0] = _normalize(start_tangent) * np.linalg.norm(m[0])
    if (not closed) and end_tangent is not None:
        m[-1] = _normalize(end_tangent) * np.linalg.norm(m[-1])

    # -- create coordinate‑wise Hermite splines ------------------------------
    xs = CubicHermiteSpline(t, points[:, 0], m[:, 0])
    ys = CubicHermiteSpline(t, points[:, 1], m[:, 1])
    zs = CubicHermiteSpline(t, points[:, 2], m[:, 2])

    n_seg = len(points) - 1 if not closed else len(points)
    n_eval = n_seg * samples_per_segment + 1
    t_eval = np.linspace(t[0], t[-1], n_eval)

    curve_xyz = np.column_stack([xs(t_eval), ys(t_eval), zs(t_eval)])

    # -- wrap into PyVista PolyData ------------------------------------------
    n_pts = curve_xyz.shape[0]
    poly_line = np.hstack(([n_pts], np.arange(n_pts))).astype(np.int64)

    pd = pv.PolyData(curve_xyz)
    pd.lines = poly_line
    return pd


# -----------------------------------------------------------------------------
#   Skeleton model – dynamic & efficient recomputation
# -----------------------------------------------------------------------------

class SkeletonModel:
    """Mutable Circle‑of‑Willis template skeleton.

    *Knots* (explicit artery control points) can be moved interactively; all
    affected splines are lazily recomputed.  Endpoint clamp directions are
    extracted **from the current splines** of connected arteries, allowing
    smooth C¹ joins even when communicating branches attach to *mid‑points* of
    other vessels (including outlet arteries).
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

        # store control points as float arrays
        self._points: Dict[str, np.ndarray] = {
            **{n: np.asarray(p, float) for n, p in inlet.items()},
            **{n: np.asarray(p, float) for n, p in outlet.items()},
            **{n: np.asarray(p, float) for n, p in communicating.items()},
        }

        # spline cache {artery → PolyData}
        self._splines: Dict[str, pv.PolyData] = {}

        # mapping junction‑node‑id → List[(artery, knot‑index)]
        self._junction_map: Dict[int, List[Tuple[str, int]]] = {}

        self._rebuild_junction_index()
        self._recompute_all()

    # ------------------------------------------------------------------
    #   Public API
    # ------------------------------------------------------------------

    def move_knot(self, artery: str, index: int, new_xyz: Tuple[float, float, float]) -> Set[str]:
        """Move one explicit control point and recompute affected splines.

        Returns
        -------
        set[str]
            Artery names that were recomputed.
        """
        if artery not in self._points:
            raise KeyError(f"Unknown artery '{artery}'.")
        pts = self._points[artery]
        if not (0 <= index < len(pts)):
            raise IndexError("knot index out of range")

        old_xyz = pts[index].copy()
        pts[index] = np.asarray(new_xyz, float)

        # rebuild junction graph (cheap)
        self._rebuild_junction_index()

        # arteries influenced by either old or new position
        affected: Set[str] = {artery}
        affected |= self._arteries_sharing_point(old_xyz)
        affected |= self._arteries_sharing_point(new_xyz)

        # --- two‑pass recompute: parents first, then children ---------------
        # pass 1: recompute every affected artery (parents change tangents)
        for art in affected:
            self._splines[art] = self._compute_spline(art)
        # pass 2: redo – ensures children pick up any updated parent tangents
        for art in affected:
            self._splines[art] = self._compute_spline(art)

        return affected

    def get_spline(self, artery: str) -> pv.PolyData:
        return self._splines[artery]

    def all_splines(self) -> Dict[str, pv.PolyData]:
        return self._splines

    # ------------------------------------------------------------------
    #   Internal helpers
    # ------------------------------------------------------------------

    # ---- junction indexing --------------------------------------------------

    def _rebuild_junction_index(self) -> None:
        """Index *all* control‑point coordinates shared across arteries."""
        self._junction_map.clear()
        representatives: List[np.ndarray] = []  # 1 coord per distinct node

        def _find_node_id(xyz: np.ndarray) -> int:
            for nid, rep in enumerate(representatives):
                if np.linalg.norm(rep - xyz) < self.tol:
                    return nid
            return -1

        for art, pts in self._points.items():
            for idx, p in enumerate(pts):
                nid = _find_node_id(p)
                if nid == -1:
                    nid = len(representatives)
                    representatives.append(p.copy())
                self._junction_map.setdefault(nid, []).append((art, idx))

    def _arteries_sharing_point(self, xyz: np.ndarray) -> Set[str]:
        for members in self._junction_map.values():
            rep = self._points[members[0][0]][members[0][1]]
            if np.linalg.norm(rep - xyz) < self.tol:
                return {art for art, _ in members}
        return set()

    # ---- geometry utilities -------------------------------------------------

    @staticmethod
    def _spline_tangent(poly: pv.PolyData, xyz: np.ndarray) -> Optional[np.ndarray]:
        """Approximate tangent of *poly* at (closest) point *xyz*."""
        pts = poly.points
        dists = np.linalg.norm(pts - xyz, axis=1)
        idx = dists.argmin()
        if dists[idx] > _TOL:
            return None  # no reasonable match
        if idx == 0:
            tan = pts[1] - pts[0]
        elif idx == len(pts) - 1:
            tan = pts[-1] - pts[-2]
        else:
            tan = pts[idx + 1] - pts[idx - 1]
        return tan if np.linalg.norm(tan) > 0 else None

    def _matching_tangent(self, child: str, xyz: np.ndarray) -> Optional[np.ndarray]:
        """Search arteries sharing *xyz* (excluding *child*) and return spline tangent."""
        for members in self._junction_map.values():
            rep_xyz = self._points[members[0][0]][members[0][1]]
            if np.linalg.norm(rep_xyz - xyz) >= self.tol:
                continue
            for art, _ in members:
                if art == child:
                    continue
                parent_spline = self._splines.get(art)
                if parent_spline is None:
                    continue  # parent not computed yet
                tan = self._spline_tangent(parent_spline, xyz)
                if tan is not None:
                    return tan
        return None

    # ---- spline (re)generation ---------------------------------------------

    def _compute_spline(self, art: str) -> pv.PolyData:
        pts = self._points[art]
        start_tan = self._matching_tangent(art, pts[0])
        end_tan = self._matching_tangent(art, pts[-1])
        return catmull_rom_spline_polydata(
            pts.copy(),
            samples_per_segment=self.samples_per_segment,
            start_tangent=start_tan,
            end_tangent=end_tan,
        )

    def _recompute_all(self) -> None:
        for art in self._points:
            self._splines[art] = self._compute_spline(art)
        # one additional pass so children capture fresh parent tangents
        for art in self._points:
            self._splines[art] = self._compute_spline(art)


# -----------------------------------------------------------------------------
#   Example usage
# -----------------------------------------------------------------------------

if __name__ == "__main__":
    # (same control‑point dictionaries as before)
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

    skel = SkeletonModel(inlet_arteries, outlet_arteries, communicating_arteries)

    # demonstrate dynamic adjustment
    #changed = skel.move_knot("RPCOM", 0, (-5.2, -4.1, -0.1))
    #print("Recomputed:", changed)

    # visualise
    pl = pv.Plotter()
    for poly in skel.all_splines().values():
        pl.add_mesh(poly, line_width=6, render_lines_as_tubes=True)
    pl.camera_position = 'xy'
    pl.show()
