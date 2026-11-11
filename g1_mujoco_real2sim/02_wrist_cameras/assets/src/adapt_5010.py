"""Adapt the Unitree G1 D405 wrist mount (M4010 wrist) to the 5010 wrist.

The original xr_teleoperate mount clamps a ring onto the wrist yaw housing
next to the hand. On the M4010 wrist that housing is a ~Ø61 mm circle with a
single flat; the 5010 wrist (unitree_ros ``*_5010.STL`` meshes, used by the
``g1_29dof_mode_15/16/18`` URDFs) has flats on both sides and a fuller top
cap, so the original ring does not seat.

This script keeps the original solids and only re-cuts the wrist bore:

1. fill the old bore inside the 10 mm clamping band (on the side that gains
   the new flat), then
2. cut a new bore traced from the 5010 yaw-link mesh plus ``CLEARANCE``.

Coordinates (ring frame of the original STEP files): the ring axis is the
bracket Z axis / clamp Y axis; ring "u" is X in both parts; ring "v" is
bracket Y / clamp Z (v > 0 = camera side, which sits on the yaw link's top
cap). Wrist yaw-link frame mapping: u = -y_yaw (left mesh), v = z_yaw.

Usage (from this directory, with build123d, trimesh and numpy installed):

    python adapt_5010.py ../d405-wrist-mount . [clearance_mm]
"""

import sys
import tempfile
import urllib.request
from pathlib import Path

import numpy as np
import trimesh
from build123d import (
    Align,
    Box,
    Cylinder,
    Face,
    Pos,
    Spline,
    Vector,
    Wire,
    export_step,
    export_stl,
    extrude,
    import_step,
    mirror,
    Plane,
)

MESH_URL = (
    "https://raw.githubusercontent.com/unitreerobotics/unitree_ros/master/"
    "robots/g1_description/meshes/left_wrist_yaw_link_5010.STL"
)

CLEARANCE = 0.25  # radial clearance over the mesh envelope [mm]

# Axial layout of the original parts.
BAND_Z = (15.0, 25.0)  # clamping band along the right bracket's Z axis
CLAMP_Y = (113.0, 123.0)  # same band along the clamp's Y axis
SPLIT = 0.5  # bracket keeps v >= +0.5, clamp keeps v <= -0.5

# Yaw-link X range (hand flange is at x = 41.5) covered by the band (on the
# flange, x ~ 28..37) and the bracket's top "tongue" (on the top cap toward
# the elbow), taken generously. Both see nearly the same outline (< 0.4 mm
# apart), so a single bore profile is used for the whole saddle.
WRIST_X = (-8.0, 40.0)
SADDLE_Z = (-40.0, BAND_Z[1] + 2.0)  # right bracket, past both saddle ends


def yaw_mesh(cache: Path) -> trimesh.Trimesh:
    path = cache / "left_wrist_yaw_link_5010.STL"
    if not path.exists():
        cache.mkdir(parents=True, exist_ok=True)
        urllib.request.urlretrieve(MESH_URL, path)
    mesh = trimesh.load(path)
    mesh.apply_scale(1000.0)  # m -> mm
    return mesh


def envelope(mesh: trimesh.Trimesh, x_range, step_deg=1.0) -> np.ndarray:
    """Max radius of the mesh cross-sections per angle, in ring (u, v)."""
    pts = []
    for x in np.arange(x_range[0], x_range[1] + 1e-9, 0.5):
        sec = mesh.section(plane_origin=[x, 0, 0], plane_normal=[1, 0, 0])
        if sec is None:
            continue
        for ent in sec.entities:
            seg = sec.vertices[ent.points][:, 1:]  # (y, z)
            for a, b in zip(seg[:-1], seg[1:]):  # densify long (flat) edges
                n = max(2, int(np.linalg.norm(b - a) / 0.2) + 1)
                pts.append(a + (b - a) * np.linspace(0, 1, n)[:, None])
    yz = np.vstack(pts)
    u, v = -yz[:, 0], yz[:, 1]
    ang = np.degrees(np.arctan2(v, u)) % 360.0
    rad = np.hypot(u, v)
    nb = int(round(360 / step_deg))
    idx = (ang / step_deg).astype(int) % nb
    env = np.full(nb, np.nan)
    np.fmax.at(env, idx, rad)
    good = ~np.isnan(env)
    k = np.arange(nb)
    env = np.interp(k, k[good], env[good], period=nb)
    return env


def bore_points(env: np.ndarray, clearance: float):
    nb = len(env)
    w = max(1, int(round(2 * nb / 360)))  # +-2 deg
    stack = np.stack([np.roll(env, s) for s in range(-w, w + 1)])
    smooth = stack.max(axis=0)  # never go inside the mesh
    smooth = np.stack([np.roll(smooth, s) for s in range(-w, w + 1)]).mean(0)
    r = smooth + clearance
    t = np.radians(np.arange(nb) * 360 / nb)
    return np.c_[r * np.cos(t), r * np.sin(t)][::2]  # 2 deg spline knots


def prism(uv, axis: str, lo: float, hi: float):
    """Extrude the closed (u, v) bore curve along the part's ring axis."""
    if axis == "z":  # bracket: (u, v, axial) = (x, y, z)
        p3 = [Vector(u, v, lo) for u, v in uv]
        d = Vector(0, 0, 1)
    else:  # clamp: (u, axial, v) = (x, y, z)
        p3 = [Vector(u, lo, v) for u, v in uv]
        d = Vector(0, 1, 0)
    face = Face(Wire([Spline(*p3, periodic=True)]))
    return extrude(face, amount=hi - lo, dir=d)


def band_fill(axis: str, lo: float, hi: float):
    """Solid filling the old bore on the u < 0 half of the part's band.

    Limited to u < 0 so the M3 tab hole at u = +27.9 is left untouched; the
    u > 0 side already has a flat in the original part.
    """
    r = 31.5
    mn = (Align.MIN, Align.MIN, Align.MIN)
    if axis == "z":  # bracket half: v = y >= SPLIT
        cyl = Pos(0, 0, lo) * Cylinder(r, hi - lo, align=(Align.CENTER, Align.CENTER, Align.MIN))
        keep = Pos(-r, SPLIT, lo) * Box(r, r, hi - lo, align=mn)
    else:  # clamp half: v = z <= -SPLIT
        cyl = Pos(0, lo, 0) * Cylinder(r, hi - lo, rotation=(-90, 0, 0),
                                        align=(Align.CENTER, Align.CENTER, Align.MIN))
        keep = Pos(-r, lo, -r) * Box(r, hi - lo, r - SPLIT, align=mn)
    return cyl & keep


def main():
    src, out = Path(sys.argv[1]), Path(sys.argv[2])
    clearance = float(sys.argv[3]) if len(sys.argv) > 3 else CLEARANCE
    out.mkdir(parents=True, exist_ok=True)
    mesh = yaw_mesh(Path(tempfile.gettempdir()) / "unitree-g1-5010-meshes")
    bore = bore_points(envelope(mesh, WRIST_X), clearance)

    # Right bracket; the original left bracket is its mirror image across
    # the XY plane, so mirror the result instead of cutting it twice.
    right = import_step(str(src / "unitree-g1-d405-wrist-mount-right.step"))
    right = checked_cut(right + band_fill("z", *BAND_Z), prism(bore, "z", *SADDLE_Z), "right")
    save(right, out / "unitree-g1-d405-wrist-mount-5010-right")
    save(mirror(right, Plane.XY), out / "unitree-g1-d405-wrist-mount-5010-left")

    clamp = import_step(str(src / "unitree-g1-d405-wrist-ring-clamp.step"))
    cutter = prism(bore, "y", CLAMP_Y[0] - 1, CLAMP_Y[1] + 1)
    clamp = checked_cut(clamp + band_fill("y", *CLAMP_Y), cutter, "clamp")
    save(clamp, out / "unitree-g1-d405-wrist-ring-clamp-5010")


def checked_cut(solid, cutter, label):
    """Subtract and make sure OCC actually removed the overlap.

    OCC booleans can fail silently (returning the input unchanged), e.g. on
    coincident spline faces, so verify the result no longer meets the cutter.
    """
    overlap = solid & cutter
    before = overlap.volume if overlap is not None else 0.0
    result = solid - cutter
    left = result & cutter
    left = left.volume if left is not None else 0.0
    if before < 1.0 or left > 1e-3 or len(result.solids()) != 1 or not result.is_valid:
        raise RuntimeError(f"{label}: cut failed (overlap {before:.1f} -> {left:.3f} mm^3)")
    print(f"{label}: removed {before:.0f} mm^3, volume {result.volume:.0f} mm^3")
    return result


def save(solid, stem: Path):
    export_step(solid, str(stem.with_suffix(".step")))
    export_stl(solid, str(stem.with_suffix(".stl")), tolerance=0.01, angular_tolerance=0.1)


if __name__ == "__main__":
    main()
