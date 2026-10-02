"""
stl_to_mesh.py: read an STL file and build a 2D (surface) or 3D (volume) mesh with Gmsh.
The output format is picked from the extension of the output file (.unv, .bdf, .msh, .vtk, ...).

Requires: python -m pip install gmsh   (pymeshlab is optional, see --surface)

Examples
--------
  # Tetrahedral volume mesh, 2 to 10 mm elements, UNV output in meters
  python stl_to_mesh.py head.stl head.unv --dim 3 --min 2 --max 10 --elem tet --scale 0.001

  # Quad surface mesh, second order elements
  python stl_to_mesh.py head.stl head_surf.unv --dim 2 --min 3 --max 8 --elem quad --order 2

  # From Python
  from stl_to_mesh import stl_to_mesh
  stl_to_mesh("head.stl", "head.unv", dim=3, min_size=2, max_size=10, elem="tet", scale=1e-3)

Surface modes (--surface)
-------------------------
  auto       "keep" in 3D, "isotropic" in 2D
  keep       use the STL triangles as they are (fast). In 3D the tets are built from the STL
             boundary and grow toward --max inside. In 2D the STL is just converted/exported,
             so sizes are not applied.
  isotropic  isotropic remesh with pymeshlab, target length = --max (adaptive if
             --curvature > 0), then same as "keep". --min is unused in this mode.
  gmsh       let Gmsh reparametrize the surface (classifySurfaces + createGeometry). Honors
             --min/--max but can be very slow, or hang, on real-world STL files.

Sizes
-----
  max_size is the largest element size (flat areas, volume interior). min_size is reached where
  curvature is high (only if curvature > 0). Both are in STL units; `scale` only rescales the
  written mesh (e.g. 0.001 for an STL in mm and a mesh in m).

Element types
-------------
  dim 2: "tri", "quad" (mostly quads), "quad-full" (all quads, by subdivision)
  dim 3: "tet", "hex" (all hexes by subdividing tets, experimental, mediocre quality)
  Sizes are halved automatically for the subdivision types ("quad-full", "hex").
"""

import argparse
import math
import os
import re
import struct
import sys
import tempfile
import time

ELEMS_2D = ("tri", "quad", "quad-full")
ELEMS_3D = ("tet", "hex")

_T0 = time.time()


def _log(msg):
    print("[%6.1fs] %s" % (time.time() - _T0, msg), flush=True)


# ---------------------------------------------------------------------
#  STL reading / writing / cleanup
# ---------------------------------------------------------------------
def _stl_dtype():
    import numpy as np
    return np.dtype([("n", "<f4", (3,)), ("v", "<f4", (3, 3)), ("a", "<u2")])


def _read_stl(path):
    """Read an ASCII or binary STL -> array of shape (n_triangles, 3, 3)."""
    import numpy as np
    with open(path, "rb") as fh:
        data = fh.read()
    if len(data) >= 84:
        n = struct.unpack("<I", data[80:84])[0]
        if len(data) == 84 + 50 * n:
            arr = np.frombuffer(data, dtype=_stl_dtype(), count=n, offset=84)
            return arr["v"].astype(np.float64)
    txt = data.decode("utf-8", errors="ignore")
    nums = re.findall(r"vertex\s+([-+0-9.eE]+)\s+([-+0-9.eE]+)\s+([-+0-9.eE]+)", txt)
    if not nums:
        raise ValueError("Could not read STL (neither binary nor valid ASCII): " + path)
    return np.array(nums, dtype=np.float64).reshape(-1, 3, 3)


def _write_stl(tris, path):
    import numpy as np
    arr = np.zeros(len(tris), dtype=_stl_dtype())
    normals = np.cross(tris[:, 1] - tris[:, 0], tris[:, 2] - tris[:, 0])
    norm = np.linalg.norm(normals, axis=1, keepdims=True)
    norm[norm == 0] = 1.0
    arr["n"] = normals / norm
    arr["v"] = tris
    with open(path, "wb") as fh:
        fh.write(b"\0" * 80)
        fh.write(struct.pack("<I", len(tris)))
        fh.write(arr.tobytes())


def _topology(tris):
    """Weld vertices by coordinates and count open edges and non-manifold edges."""
    import numpy as np
    pts = tris.reshape(-1, 3).astype(np.float32).astype(np.float64)   # same precision as the written STL
    diag = float(np.linalg.norm(pts.max(0) - pts.min(0))) or 1.0
    key = np.round(pts / (diag * 1e-7)).astype(np.int64)
    _, idx = np.unique(key, axis=0, return_inverse=True)
    idx = np.asarray(idx).reshape(-1, 3)
    e = np.concatenate([idx[:, [0, 1]], idx[:, [1, 2]], idx[:, [2, 0]]])
    e.sort(axis=1)
    _, cnt = np.unique(e, axis=0, return_counts=True)
    return {"vertices": int(idx.max()) + 1, "edges": int(len(cnt)),
            "boundary": int((cnt == 1).sum()),       # used by a single triangle (holes)
            "non_manifold": int((cnt > 2).sum())}    # shared by more than 2 triangles


def _clean_stl(tris, min_tris):
    """Weld near-identical vertices, drop degenerate triangles, then subdivide
    (edge midpoints) until there are at least min_tris triangles.

    Gmsh loops for a very long time on tiny STLs (a rectangle = 2 triangles)."""
    import numpy as np
    all_pts = tris.reshape(-1, 3)
    diag = float(np.linalg.norm(all_pts.max(0) - all_pts.min(0)))
    tol = diag * 1e-7
    pts = np.round(tris / tol) * tol
    area = 0.5 * np.linalg.norm(np.cross(pts[:, 1] - pts[:, 0], pts[:, 2] - pts[:, 0]), axis=1)
    pts = pts[area > diag ** 2 * 1e-14]
    if len(pts) == 0:
        raise ValueError("All STL triangles are degenerate.")
    while len(pts) < min_tris:
        p0, p1, p2 = pts[:, 0], pts[:, 1], pts[:, 2]
        m01, m12, m20 = (p0 + p1) / 2, (p1 + p2) / 2, (p2 + p0) / 2
        pts = np.concatenate([np.stack([p0, m01, m20], 1), np.stack([m01, p1, m12], 1),
                              np.stack([m20, m12, p2], 1), np.stack([m01, m12, m20], 1)])
    return pts


# ---------------------------------------------------------------------
#  pymeshlab helpers
# ---------------------------------------------------------------------
def _meshlab_to_tris(ms):
    import numpy as np
    m = ms.current_mesh()
    v = np.asarray(m.vertex_matrix(), dtype=np.float64)
    f = np.asarray(m.face_matrix(), dtype=np.int64)
    return v[f]


def _repair(tris, workdir, close_holes):
    """Remove duplicates and non-manifold edges with pymeshlab, if available."""
    try:
        import pymeshlab
    except ImportError:
        print("  (pymeshlab not installed, cannot repair)")
        return tris
    src = os.path.join(workdir, "to_repair.stl")
    _write_stl(tris, src)
    ms = pymeshlab.MeshSet()
    ms.load_new_mesh(src)

    def apply(names, **kw):
        # filter names changed between pymeshlab versions: use the first one that exists
        for name in names:
            fn = getattr(ms, name, None)
            if fn is not None:
                try:
                    fn(**kw)
                except Exception as e:
                    print("  repair step '%s' skipped (%s)" % (name, e))
                return

    apply(["meshing_remove_duplicate_vertices", "remove_duplicate_vertices"])
    apply(["meshing_remove_duplicate_faces", "remove_duplicate_faces"])
    apply(["meshing_repair_non_manifold_edges", "repair_non_manifold_edges_by_removing_faces"])
    apply(["meshing_remove_unreferenced_vertices", "remove_unreferenced_vertices"])
    if close_holes:
        apply(["meshing_close_holes", "close_holes"], maxholesize=50)
    return _meshlab_to_tris(ms)


def _remesh_isotropic(tris, workdir, target, angle, adaptive):
    """Isotropic surface remesh with pymeshlab (ImportError is handled by the caller)."""
    import pymeshlab
    src = os.path.join(workdir, "before_remesh.stl")
    _write_stl(tris, src)
    ms = pymeshlab.MeshSet()
    ms.load_new_mesh(src)
    value = getattr(pymeshlab, "PureValue", None) or getattr(pymeshlab, "AbsoluteValue")
    remesh = (getattr(ms, "meshing_isotropic_explicit_remeshing", None)
              or getattr(ms, "remeshing_isotropic_explicit_remeshing"))
    remesh(iterations=5, adaptive=bool(adaptive), targetlen=value(target), featuredeg=angle)
    return _meshlab_to_tris(ms)


# ---------------------------------------------------------------------
#  Main function
# ---------------------------------------------------------------------
def stl_to_mesh(stl_file, out_file, dim=3, min_size=2.0, max_size=10.0, elem=None,
                order=1, angle=40.0, curvature=20, scale=1.0, gui=False,
                surface="auto", min_tri=None):
    """Mesh an STL file and write the result.

    stl_file  : input STL (ASCII or binary); must be watertight for dim=3
    out_file  : output file, format chosen from the extension
    dim       : 2 (surface mesh) or 3 (volume mesh)
    min_size  : minimum element size (STL units)
    max_size  : maximum element size (STL units)
    elem      : element type, defaults to "tri" in 2D and "tet" in 3D
    order     : 1 (linear) or 2 (quadratic)
    angle     : edges sharper than this (degrees) are treated as features
    curvature : elements per full turn for curvature refinement, 0 = off
    scale     : factor applied to the written coordinates (0.001: mm -> m)
    gui       : open the Gmsh window at the end
    surface   : "auto", "keep", "isotropic" or "gmsh" (see module docstring)
    min_tri   : minimum STL triangle count before handing it to Gmsh
                ("gmsh" mode only, default 500, 0 otherwise)
    """
    try:
        import gmsh
    except ImportError as e:
        raise ImportError("Module 'gmsh' not found. Install it with: python -m pip install gmsh") from e

    if dim not in (2, 3):
        raise ValueError("dim must be 2 or 3.")
    if elem is None:
        elem = "tri" if dim == 2 else "tet"
    if dim == 2 and elem not in ELEMS_2D:
        raise ValueError("For dim=2, elem must be one of %s." % (ELEMS_2D,))
    if dim == 3 and elem not in ELEMS_3D:
        raise ValueError("For dim=3, elem must be one of %s." % (ELEMS_3D,))
    if min_size <= 0 or max_size < min_size:
        raise ValueError("Need 0 < min_size <= max_size.")
    if not os.path.isfile(stl_file):
        raise FileNotFoundError(stl_file)
    if surface == "auto":
        surface = "keep" if dim == 3 else "isotropic"
    if surface not in ("keep", "isotropic", "gmsh"):
        raise ValueError("surface must be auto, keep, isotropic or gmsh.")
    if min_tri is None:
        min_tri = 500 if surface == "gmsh" else 0

    # subdivision halves the element size, so ask for twice as much up front
    f = 2.0 if elem in ("quad-full", "hex") else 1.0

    # --- read and clean the STL ---
    _log("Reading STL...")
    raw = _read_stl(stl_file)
    tris = _clean_stl(raw, min_tri)
    mn, mx = tris.reshape(-1, 3).min(0), tris.reshape(-1, 3).max(0)
    _log("STL: %d triangles read, %d after cleanup" % (len(raw), len(tris)))
    print("  bounding box: X %.4g..%.4g | Y %.4g..%.4g | Z %.4g..%.4g"
          % (mn[0], mx[0], mn[1], mx[1], mn[2], mx[2]))
    extent = mx - mn
    if extent.min() < 1e-6 * extent.max():
        print("  -> flat surface detected")
    workdir = tempfile.mkdtemp(prefix="stl_mesh_")

    def report(label, t):
        topo = _topology(t)
        _log("Topology (%s): %d vertices, %d edges, %d open edges, %d non-manifold edges"
             % (label, topo["vertices"], topo["edges"], topo["boundary"], topo["non_manifold"]))
        return topo

    def needs_repair(topo):
        return topo["non_manifold"] or (dim == 3 and topo["boundary"])

    topo = report("input STL", tris)
    if surface != "gmsh" and needs_repair(topo):
        _log("Repairing STL (pymeshlab)...")
        tris = _repair(tris, workdir, dim == 3)
        topo = report("after repair", tris)

    if surface == "isotropic":
        try:
            _log("Isotropic remesh (pymeshlab), target size = %g..." % (max_size * f))
            tris = _remesh_isotropic(tris, workdir, max_size * f, angle, curvature > 0)
            _log("Remeshed surface: %d triangles" % len(tris))
            topo = report("after remesh", tris)
            if needs_repair(topo):
                tris = _repair(tris, workdir, dim == 3)
                topo = report("after second repair", tris)
        except ImportError:
            print("WARNING: pymeshlab not found (python -m pip install pymeshlab). "
                  "Falling back to 'keep': the STL is used as is and sizes are NOT applied "
                  "to the surface.")
            surface = "keep"

    if dim == 3 and surface != "gmsh" and (topo["non_manifold"] or topo["boundary"]):
        raise RuntimeError(
            "The STL is not watertight or has non-manifold edges (%d open, %d non-manifold), "
            "so no volume can be built. Fix it (MeshLab, or install pymeshlab for automatic "
            "repair) or use --dim 2." % (topo["boundary"], topo["non_manifold"]))

    prepared = os.path.join(workdir, "prepared.stl")
    _write_stl(tris, prepared)

    gmsh.initialize()
    try:
        gmsh.option.setNumber("General.Terminal", 1)
        gmsh.model.add("stl_mesh")

        # --- geometry from the STL ---
        _log("Gmsh: reading STL...")
        gmsh.merge(prepared)
        if surface == "gmsh":
            _log("Gmsh: classifying surfaces and reparametrizing (can take very long)...")
            gmsh.model.mesh.classifySurfaces(math.radians(angle), True, True, math.pi)
            gmsh.model.mesh.createGeometry()
        # keep / isotropic skip this: Gmsh can crash on non-manifold STLs and the
        # STL is just one discrete surface anyway

        surfaces = [t for _, t in gmsh.model.getEntities(2)]
        if not surfaces:
            raise RuntimeError("No surface could be built from the STL.")
        _log("%d surface(s) found" % len(surfaces))

        if dim == 3:
            loop = gmsh.model.geo.addSurfaceLoop(surfaces)
            volume = gmsh.model.geo.addVolume([loop])
            gmsh.model.geo.synchronize()
            gmsh.model.addPhysicalGroup(3, [volume], name="VOLUME")
            gmsh.model.addPhysicalGroup(2, surfaces, name="SKIN")
        else:
            gmsh.model.addPhysicalGroup(2, surfaces, name="SURFACE")

        # --- mesh options ---
        gmsh.option.setNumber("Mesh.MeshSizeMin", min_size * f)
        gmsh.option.setNumber("Mesh.MeshSizeMax", max_size * f)
        gmsh.option.setNumber("Mesh.MeshSizeFromPoints", 0)
        gmsh.option.setNumber("Mesh.MeshSizeExtendFromBoundary", 1)
        gmsh.option.setNumber("Mesh.MeshSizeFromCurvature", curvature if curvature > 0 else 0)
        gmsh.option.setNumber("Mesh.ElementOrder", order)
        gmsh.option.setNumber("Mesh.SaveAll", 0)             # only write physical groups
        gmsh.option.setNumber("Mesh.ScalingFactor", scale)

        if elem in ("tri", "tet", "hex"):
            gmsh.option.setNumber("Mesh.Algorithm", 6)       # Frontal-Delaunay
        else:
            gmsh.option.setNumber("Mesh.Algorithm", 8)       # Frontal-Delaunay for quads
            gmsh.option.setNumber("Mesh.RecombineAll", 1)
            gmsh.option.setNumber("Mesh.RecombinationAlgorithm", 1)   # blossom
        if elem in ("tet", "hex"):
            gmsh.option.setNumber("Mesh.Algorithm3D", 1)     # Delaunay
        if elem == "hex":
            gmsh.option.setNumber("Mesh.SubdivisionAlgorithm", 2)     # all hexahedra
        if elem == "quad-full" and surface == "gmsh":
            gmsh.option.setNumber("Mesh.SubdivisionAlgorithm", 1)     # all quads

        # --- generate ---
        if dim == 3 or surface == "gmsh":
            _log("Gmsh: generating %dD mesh..." % dim)
            gmsh.model.mesh.generate(dim)
        elif elem in ("quad", "quad-full"):
            # the STL triangles already are the 2D mesh, just recombine them
            try:
                _log("Converting triangles to quads...")
                gmsh.model.mesh.recombine()
                if elem == "quad-full":
                    gmsh.option.setNumber("Mesh.SubdivisionAlgorithm", 1)
                    gmsh.model.mesh.refine()
            except Exception as e:
                print("WARNING: quad conversion failed (%s), keeping triangles." % e)
        if order == 2:
            gmsh.model.mesh.setOrder(2)

        _log("Writing %s..." % out_file)
        gmsh.write(out_file)

        # --- summary ---
        print("\nMesh written:", os.path.abspath(out_file))
        for d in range(1, dim + 1):
            for t in gmsh.model.mesh.getElementTypes(d):
                name = gmsh.model.mesh.getElementProperties(t)[0]
                n = len(gmsh.model.mesh.getElementsByType(t)[0])
                print("  dim %d: %-18s %d" % (d, name, n))
        print("  nodes: %d" % len(gmsh.model.mesh.getNodes()[0]))
        print("  output units: STL coordinates x %g" % scale)

        if gui:
            gmsh.fltk.run()
    finally:
        gmsh.finalize()


def _parse_args(argv):
    p = argparse.ArgumentParser(description="2D/3D meshing of an STL file with Gmsh (UNV, BDF, MSH... export).")
    p.add_argument("stl", help="input STL file")
    p.add_argument("output", help="output file (.unv, .bdf, .msh, .vtk, .inp...)")
    p.add_argument("--dim", type=int, choices=(2, 3), default=3, help="2 = surface, 3 = volume (default 3)")
    p.add_argument("--min", dest="min_size", type=float, required=True, help="minimum element size")
    p.add_argument("--max", dest="max_size", type=float, required=True, help="maximum element size")
    p.add_argument("--elem", choices=ELEMS_2D + ELEMS_3D, default=None,
                   help="element type (default: tri in 2D, tet in 3D)")
    p.add_argument("--order", type=int, choices=(1, 2), default=1, help="element order (default 1)")
    p.add_argument("--angle", type=float, default=40.0, help="feature edge angle in degrees (default 40)")
    p.add_argument("--curvature", type=float, default=20,
                   help="elements per turn for curvature refinement, 0 = off (default 20)")
    p.add_argument("--scale", type=float, default=1.0, help="scale factor for the written mesh (0.001: mm -> m)")
    p.add_argument("--surface", choices=("auto", "keep", "isotropic", "gmsh"), default="auto",
                   help="how to treat the STL surface (default auto: keep in 3D, isotropic in 2D)")
    p.add_argument("--min-tri", type=int, default=None,
                   help="minimum STL triangle count (gmsh mode only, default 500)")
    p.add_argument("--gui", action="store_true", help="open Gmsh at the end")
    return p.parse_args(argv)


if __name__ == "__main__":
    a = _parse_args(sys.argv[1:])
    stl_to_mesh(a.stl, a.output, dim=a.dim, min_size=a.min_size, max_size=a.max_size,
                elem=a.elem, order=a.order, angle=a.angle, curvature=a.curvature,
                scale=a.scale, gui=a.gui, surface=a.surface, min_tri=a.min_tri)
