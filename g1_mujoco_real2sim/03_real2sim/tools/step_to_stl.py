"""Convert STEP parts to STL (mm, as in the CAD) with cascadio + trimesh.

Run in a throwaway env so no project env is touched:
    uv run --with cascadio --with trimesh python 03_real2sim/tools/step_to_stl.py in.step [more.step ...]
Each output is written next to its input with a .stl extension.
"""

import os
import sys
import tempfile

import cascadio
import trimesh

for step in sys.argv[1:]:
    with tempfile.TemporaryDirectory() as tmp:
        glb = os.path.join(tmp, "part.glb")
        cascadio.step_to_glb(step, glb, tol_linear=0.05, tol_angular=0.3)
        mesh = trimesh.load(glb, force="mesh")
    # cascadio writes glTF in metres; the STEP is in mm -> keep mm to match the other bracket STLs.
    if mesh.extents.max() < 1.0:
        mesh.apply_scale(1000.0)
    out = os.path.splitext(step)[0] + ".stl"
    mesh.export(out)
    print(f"{os.path.basename(out)}: {len(mesh.faces)} faces, extents {mesh.extents.round(2)} mm, "
          f"bounds {mesh.bounds.round(2).tolist()}")
