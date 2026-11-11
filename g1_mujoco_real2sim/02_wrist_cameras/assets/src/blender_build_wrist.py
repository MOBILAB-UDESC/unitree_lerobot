"""Blender script: gloves over the G1 rubber hands + MOBILAB D405 wrist mounts, exported for MuJoCo.

Run inside Blender (MCP: exec(open(this_file).read())) or `blender -b -P blender_build_wrist.py`.
Inputs (src/): Unitree's official rubber hands and 5010 wrist yaw links (unitree_ros g1_description,
used by g1_29dof_mode_15/16/18) and MOBILAB's d405-wrist-mount-5010 bracket + ring clamp STLs (mm).
Outputs (mesh/):
  glove_{left,right}.obj      rubber hand inflated ~2.5 mm (knit glove), UV-mapped, in the hand frame
  d405_mount_{left,right}.stl bracket + ring clamp in the bracket frame (m); left = right mirrored in Z
Also lays out a `wrist_assembly` collection (yaw link + glove + mount) for visual checks.
"""

import os

import bpy
import mathutils

ASSET = os.environ.get("G1_WRIST_ASSET_DIR", os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
SRC_G1 = os.path.join(ASSET, "src")  # Unitree rubber hands (unitree_ros g1_description)
SRC_MOUNT = os.path.join(ASSET, "src")  # MOBILAB cad-models d405-wrist-mount-5010
MESH = os.path.join(ASSET, "mesh")
TEX = os.path.join(ASSET, "tex")

GLOVE_THICKNESS = 0.007  # stuffed knit work glove with rubber pads (real glove fills ~1/3 of the wrist images)
HAND_OFFSET = {"left": (0.0415, 0.003, 0.0), "right": (0.0415, -0.003, 0.0)}  # URDF *_hand_palm_joint
# MOBILAB bracket frame -> wrist yaw link (see 02_wrist_cameras/wrist_cameras.py bracket_to_yaw)
BRACKET_TO_YAW = {"right": ((0, 0, 1), (1, 0, 0), (0, 1, 0)), "left": ((0, 0, -1), (-1, 0, 0), (0, 1, 0))}
MOUNT_AXIAL_OFFSET = 0.013
# ring clamp (clamp frame: ring axis = Y, band Y 113..123) -> right bracket frame (band Z 15..25), mm
CLAMP_TO_RIGHT_BRACKET = (((1, 0, 0), (0, 0, 1), (0, -1, 0)), (0, 0, 138))


def collection(name):
    coll = bpy.data.collections.get(name) or bpy.data.collections.new(name)
    if coll.name not in bpy.context.scene.collection.children:
        bpy.context.scene.collection.children.link(coll)
    for o in list(coll.objects):
        bpy.data.objects.remove(o)
    return coll


def import_stl(path, name, coll):
    bpy.ops.wm.stl_import(filepath=path)
    o = bpy.context.active_object
    o.name = name
    for c in o.users_collection:
        c.objects.unlink(o)
    coll.objects.link(o)
    return o


def select_only(o):
    bpy.ops.object.select_all(action="DESELECT")
    o.select_set(True)
    bpy.context.view_layer.objects.active = o


def matrix(rows, t=(0, 0, 0)):
    m = mathutils.Matrix.Identity(4)
    for i in range(3):
        for j in range(3):
            m[i][j] = rows[i][j]
        m[i][3] = t[i]
    return m


def material(name, rgba, texture=None, roughness=0.9):
    mat = bpy.data.materials.get(name) or bpy.data.materials.new(name)
    mat.diffuse_color = rgba
    mat.use_nodes = True
    bsdf = next(n for n in mat.node_tree.nodes if n.type == "BSDF_PRINCIPLED")
    bsdf.inputs["Roughness"].default_value = roughness
    bsdf.inputs["Base Color"].default_value = rgba
    if texture:
        tex = mat.node_tree.nodes.new("ShaderNodeTexImage")
        tex.image = bpy.data.images.load(os.path.join(TEX, texture), check_existing=True)
        mat.node_tree.links.new(tex.outputs["Color"], bsdf.inputs["Base Color"])
    return mat


def build_glove(side, coll):
    o = import_stl(os.path.join(SRC_G1, f"{side}_rubber_hand.STL"), f"glove_{side}", coll)
    select_only(o)
    remesh = o.modifiers.new("remesh", "REMESH")
    remesh.mode, remesh.voxel_size = "VOXEL", 0.0012
    bpy.ops.object.modifier_apply(modifier=remesh.name)
    disp = o.modifiers.new("inflate", "DISPLACE")
    disp.direction, disp.strength, disp.mid_level = "NORMAL", GLOVE_THICKNESS, 0.0
    bpy.ops.object.modifier_apply(modifier=disp.name)
    smooth = o.modifiers.new("smooth", "SMOOTH")
    smooth.factor, smooth.iterations = 0.6, 6
    bpy.ops.object.modifier_apply(modifier=smooth.name)
    dec = o.modifiers.new("dec", "DECIMATE")
    dec.ratio = min(1.0, 9000 / max(1, len(o.data.polygons)))
    bpy.ops.object.modifier_apply(modifier=dec.name)
    for p in o.data.polygons:
        p.use_smooth = True
    bpy.ops.object.mode_set(mode="EDIT")
    bpy.ops.mesh.select_all(action="SELECT")
    bpy.ops.uv.smart_project(angle_limit=1.15, island_margin=0.01)
    bpy.ops.object.mode_set(mode="OBJECT")
    o.data.materials.append(material("glove_knit", (0.05, 0.05, 0.05, 1), "glove_knit.png"))
    bpy.ops.wm.obj_export(filepath=os.path.join(MESH, f"glove_{side}.obj"), export_selected_objects=True,
                          forward_axis="Y", up_axis="Z", export_materials=False, export_uv=True,
                          export_normals=True, export_triangulated_mesh=True)
    print(f"glove_{side}: {len(o.data.polygons)} faces, dims {tuple(round(v, 4) for v in o.dimensions)}")
    return o


def build_mounts(coll):
    right = import_stl(os.path.join(SRC_MOUNT, "unitree-g1-d405-wrist-mount-5010-right.stl"), "mount_right", coll)
    clamp = import_stl(os.path.join(SRC_MOUNT, "unitree-g1-d405-wrist-ring-clamp-5010.stl"), "clamp", coll)
    clamp.matrix_world = matrix(*CLAMP_TO_RIGHT_BRACKET)
    select_only(clamp)
    bpy.ops.object.transform_apply(location=True, rotation=True, scale=True)
    select_only(right)
    clamp.select_set(True)
    bpy.ops.object.join()
    right.scale = (0.001,) * 3
    bpy.ops.object.transform_apply(scale=True)
    dec = right.modifiers.new("dec", "DECIMATE")
    dec.ratio = 9000 / len(right.data.polygons)
    bpy.ops.object.modifier_apply(modifier=dec.name)
    bpy.ops.wm.stl_export(filepath=os.path.join(MESH, "d405_mount_right.stl"), export_selected_objects=True)
    # Left bracket = right bracket mirrored across the bracket XY plane (MOBILAB's definition).
    left = right.copy()
    left.data = right.data.copy()
    left.name = "mount_left"
    coll.objects.link(left)
    left.scale = (1, 1, -1)
    select_only(left)
    bpy.ops.object.transform_apply(scale=True)
    bpy.ops.object.mode_set(mode="EDIT")
    bpy.ops.mesh.select_all(action="SELECT")
    bpy.ops.mesh.normals_make_consistent(inside=False)
    bpy.ops.object.mode_set(mode="OBJECT")
    bpy.ops.wm.stl_export(filepath=os.path.join(MESH, "d405_mount_left.stl"), export_selected_objects=True)
    print("mounts:", len(right.data.polygons), "faces each")
    return {"left": left, "right": right}


def assemble(gloves, mounts, coll):
    silver = material("g1_silver", (0.75, 0.75, 0.77, 1), roughness=0.4)
    pla = material("pla_slate", (0.2, 0.22, 0.27, 1), roughness=0.6)
    for side, y0 in (("left", 0.12), ("right", -0.12)):
        root = bpy.data.objects.new(f"{side}_wrist_yaw_frame", None)
        coll.objects.link(root)
        root.location = (0, y0, 0)
        yaw = import_stl(os.path.join(MESH, f"{side}_wrist_yaw_link_5010.STL"), f"{side}_yaw_link_5010", coll)
        yaw.data.materials.append(silver)
        yaw.parent = root
        g = gloves[side]
        g.parent, g.location = root, HAND_OFFSET[side]
        m = mounts[side]
        m.data.materials.clear()
        m.data.materials.append(pla)
        m.parent = root
        m.matrix_local = matrix(BRACKET_TO_YAW[side], (MOUNT_AXIAL_OFFSET, 0, 0))


def main():
    for name in ("wrist_assembly", "mount_build"):
        collection(name)
    build = collection("wrist_build")
    gloves = {s: build_glove(s, build) for s in ("left", "right")}
    mounts = build_mounts(build)
    assemble(gloves, mounts, build)


main()
