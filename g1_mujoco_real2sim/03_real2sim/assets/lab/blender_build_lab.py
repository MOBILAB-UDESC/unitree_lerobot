"""Blender script: build the real2sim lab props and export them as OBJ for MuJoCo.

Run inside Blender (MCP execute_blender_code: exec(open(this_file).read())) or
`blender -b -P blender_build_lab.py`. Units are metres, Z up, and each object is modelled in the
frame MuJoCo uses for it (origin noted per object). Exports go to ../lab/mesh/*.obj, one mesh per
material because MuJoCo meshes carry a single material.

Real dimensions (cm): desk 125 x 45 x 74 high; cardboard box 49 x 33 x 18; bubble-wrapped box
23 x 23 x 27. Desk depth runs along X (robot side at -X), length along Y.
"""

import math
import os

import bmesh
import bpy
from mathutils import Vector, noise

HERE = os.path.dirname(os.path.abspath(bpy.path.abspath("//"))) if bpy.data.filepath else None
ASSET_DIR = os.environ.get("G1_LAB_ASSET_DIR", os.path.dirname(os.path.abspath(__file__)))
MESH_DIR = os.path.join(ASSET_DIR, "mesh")
TEX_DIR = os.path.join(ASSET_DIR, "tex")

DESK_L, DESK_D, DESK_H, TOP_T = 1.25, 0.45, 0.74, 0.025
BOX_L, BOX_W, BOX_H, WALL = 0.49, 0.33, 0.18, 0.004
WRAP_W, WRAP_H = 0.23, 0.27


# ----------------------------------------------------------------------------------------------
def reset_scene():
    bpy.ops.object.select_all(action="SELECT")
    bpy.ops.object.delete()
    for block in (bpy.data.meshes, bpy.data.materials, bpy.data.images):
        for item in list(block):
            if item.users == 0:
                block.remove(item)
    bpy.context.scene.unit_settings.system = "METRIC"


def material(name, texture=None, color=(0.5, 0.5, 0.5, 1), roughness=0.6, metallic=0.0, scale=1.0):
    mat = bpy.data.materials.new(name)
    mat.use_nodes = True
    nodes = mat.node_tree.nodes
    bsdf = next(n for n in nodes if n.type == "BSDF_PRINCIPLED")
    bsdf.inputs["Roughness"].default_value = roughness
    bsdf.inputs["Metallic"].default_value = metallic
    if texture:
        img = bpy.data.images.load(os.path.join(TEX_DIR, texture), check_existing=True)
        tex = nodes.new("ShaderNodeTexImage")
        tex.image = img
        mapping = nodes.new("ShaderNodeMapping")
        mapping.inputs["Scale"].default_value = (scale, scale, scale)
        coord = nodes.new("ShaderNodeTexCoord")
        mat.node_tree.links.new(coord.outputs["UV"], mapping.inputs["Vector"])
        mat.node_tree.links.new(mapping.outputs["Vector"], tex.inputs["Vector"])
        mat.node_tree.links.new(tex.outputs["Color"], bsdf.inputs["Base Color"])
    else:
        bsdf.inputs["Base Color"].default_value = color
    return mat


def box(name, size, center, mat=None, bevel=0.0, segments=3):
    bpy.ops.mesh.primitive_cube_add(size=1, location=center)
    obj = bpy.context.active_object
    obj.name = name
    obj.scale = size
    bpy.ops.object.transform_apply(scale=True)
    if bevel:
        mod = obj.modifiers.new("bevel", "BEVEL")
        mod.width = bevel
        mod.segments = segments
        mod.limit_method = "NONE"
        bpy.ops.object.modifier_apply(modifier=mod.name)
    if mat:
        obj.data.materials.append(mat)
    return obj


def cube_uv(obj, size_m):
    """World-scale box projection: one texture tile per size_m metres."""
    bpy.context.view_layer.objects.active = obj
    obj.select_set(True)
    bpy.ops.object.mode_set(mode="EDIT")
    bpy.ops.mesh.select_all(action="SELECT")
    bpy.ops.uv.cube_project(cube_size=size_m, correct_aspect=True, scale_to_bounds=False)
    bpy.ops.object.mode_set(mode="OBJECT")
    obj.select_set(False)


def join(name, objs):
    bpy.ops.object.select_all(action="DESELECT")
    for o in objs:
        o.select_set(True)
    bpy.context.view_layer.objects.active = objs[0]
    bpy.ops.object.join()
    obj = bpy.context.active_object
    obj.name = name
    return obj


# ----------------------------------------------------------------------------------------------
def build_desk():
    """Origin: floor, under the centre of the desk top. Robot side is -X."""
    m_top = material("desk_top", "desk_top.png", roughness=0.45)
    m_panel = material("desk_panel", "desk_panel.png", roughness=0.55)
    m_leg = material("desk_leg", "desk_leg.png", roughness=0.35, metallic=0.4)

    top = box("desk_top", (DESK_D, DESK_L, TOP_T), (0, 0, DESK_H - TOP_T / 2), m_top, bevel=0.008)
    cube_uv(top, 0.6)

    # Modesty panel under the top on the far side (away from the robot), as in photos 1/3.
    panel_h = 0.27
    panel = box("desk_panel", (0.016, DESK_L - 0.12, panel_h),
                (DESK_D / 2 - 0.04, 0, DESK_H - TOP_T - panel_h / 2), m_panel, bevel=0.003)
    cube_uv(panel, 0.6)

    # C-shaped side frames: post near the far edge, foot bar and top rail along the depth.
    legs = []
    for side in (-1, 1):
        y = side * (DESK_L / 2 - 0.05)
        post_x = DESK_D / 2 - 0.07
        legs.append(box("post", (0.06, 0.03, DESK_H - TOP_T - 0.04), (post_x, y, (DESK_H - TOP_T) / 2 + 0.02), bevel=0.004))
        legs.append(box("foot", (DESK_D - 0.02, 0.05, 0.04), (0, y, 0.02), bevel=0.006))
        legs.append(box("rail", (DESK_D - 0.06, 0.03, 0.03), (0, y, DESK_H - TOP_T - 0.015), bevel=0.003))
        for fx in (-1, 1):  # rubber feet caps
            legs.append(box("cap", (0.035, 0.055, 0.012), (fx * (DESK_D / 2 - 0.03), y, 0.006), bevel=0.003))
    leg = join("desk_legs", legs)
    leg.data.materials.clear()
    leg.data.materials.append(m_leg)
    cube_uv(leg, 0.3)
    return [top, panel, leg]


def build_cardboard_box():
    """Open-top box, origin at the centre of its bottom face. Long side along Y."""
    m_out = material("cardboard_out", "cardboard_out.png", roughness=0.9)
    m_in = material("cardboard_in", "cardboard_in.png", roughness=0.9)

    def shell(name, l, w, h, z0, inward):
        bm = bmesh.new()
        x, y = w / 2, l / 2
        v = [bm.verts.new(p) for p in [(-x, -y, z0), (x, -y, z0), (x, y, z0), (-x, y, z0),
                                        (-x, -y, z0 + h), (x, -y, z0 + h), (x, y, z0 + h), (-x, y, z0 + h)]]
        faces = [(0, 3, 2, 1), (0, 1, 5, 4), (1, 2, 6, 5), (2, 3, 7, 6), (3, 0, 4, 7)]  # bottom + 4 walls, outward
        for f in faces:
            bm.faces.new([v[i] for i in (f if not inward else f[::-1])])
        mesh = bpy.data.meshes.new(name)
        bm.to_mesh(mesh)
        obj = bpy.data.objects.new(name, mesh)
        bpy.context.collection.objects.link(obj)
        return obj

    outer = shell("cardboard_outer", BOX_L, BOX_W, BOX_H, 0.0, inward=False)
    inner = shell("cardboard_inner", BOX_L - 2 * WALL, BOX_W - 2 * WALL, BOX_H - WALL, WALL, inward=True)
    # Rim: thin top band joining outer and inner walls, part of the inner (lighter edge) mesh.
    rim = box("rim_tmp", (BOX_W, BOX_L, 0.002), (0, 0, BOX_H - 0.001))
    hole = box("hole_tmp", (BOX_W - 2 * WALL, BOX_L - 2 * WALL, 0.01), (0, 0, BOX_H - 0.001))
    mod = rim.modifiers.new("cut", "BOOLEAN")
    mod.operation, mod.object = "DIFFERENCE", hole
    bpy.context.view_layer.objects.active = rim
    bpy.ops.object.modifier_apply(modifier=mod.name)
    bpy.data.objects.remove(hole)
    inner = join("cardboard_inner", [inner, rim])
    outer.data.materials.append(m_out)
    inner.data.materials.append(m_in)
    cube_uv(outer, 0.5)
    cube_uv(inner, 0.5)
    return [outer, inner]


def build_wrapped_box(seed=3):
    """Bubble-wrapped box, origin at the centre of its bottom face. Lumpy, bulging shell."""
    m = material("bubble_wrap", "bubble_wrap.png", roughness=0.25)
    bpy.ops.mesh.primitive_cube_add(size=1, location=(0, 0, WRAP_H / 2))
    obj = bpy.context.active_object
    obj.name = "wrapped_box"
    obj.scale = (WRAP_W, WRAP_W, WRAP_H)
    bpy.ops.object.transform_apply(scale=True)
    sub = obj.modifiers.new("sub", "SUBSURF")
    sub.levels = 5
    sub.subdivision_type = "SIMPLE"
    bpy.ops.object.modifier_apply(modifier=sub.name)

    bm = bmesh.new()
    bm.from_mesh(obj.data)
    centre = Vector((0, 0, WRAP_H / 2))
    off = Vector((seed * 1.7, seed * 2.3, seed * 0.9))
    for v in bm.verts:
        p = v.co.copy()
        d = (p - centre)
        # Rounded corners: pull towards a superellipsoid, then wrinkle with multi-scale noise.
        n_dir = d.normalized()
        corner = max(abs(d.x) / (WRAP_W / 2), abs(d.y) / (WRAP_W / 2), abs(d.z) / (WRAP_H / 2))
        shrink = 0.016 * max(0.0, (abs(d.x) / (WRAP_W / 2)) * (abs(d.y) / (WRAP_W / 2)) +
                              (abs(d.x) / (WRAP_W / 2)) * (abs(d.z) / (WRAP_H / 2)) * 0.6 +
                              (abs(d.y) / (WRAP_W / 2)) * (abs(d.z) / (WRAP_H / 2)) * 0.6 - 0.35)
        lump = 0.012 * noise.noise(p * 9 + off) + 0.006 * noise.noise(p * 23 + off) + 0.0025 * noise.noise(p * 60 + off)
        bulge = 0.010 * (1 - abs(d.z) / (WRAP_H / 2)) ** 2  # wrap is puffier around the middle
        taper = 0.06 * max(0.0, d.z / (WRAP_H / 2))  # gathered wrap makes the top a bit narrower
        v.co = p - n_dir * shrink + n_dir * (lump + bulge) * (1 if corner > 0.5 else 0.5)
        v.co.x -= d.x * taper
        v.co.y -= d.y * taper
        if p.z < 0.003:  # keep a flat base so it sits on the desk
            v.co.z = 0.0
    bm.to_mesh(obj.data)
    bm.free()
    for poly in obj.data.polygons:
        poly.use_smooth = True
    obj.data.materials.append(m)
    bpy.context.view_layer.objects.active = obj
    obj.select_set(True)
    bpy.ops.object.mode_set(mode="EDIT")
    bpy.ops.mesh.select_all(action="SELECT")
    bpy.ops.uv.smart_project(angle_limit=math.radians(66), island_margin=0.02)
    bpy.ops.object.mode_set(mode="OBJECT")
    obj.select_set(False)
    return [obj]


def export(objs):
    os.makedirs(MESH_DIR, exist_ok=True)
    for obj in objs:
        bpy.ops.object.select_all(action="DESELECT")
        obj.select_set(True)
        bpy.context.view_layer.objects.active = obj
        bpy.ops.wm.obj_export(
            filepath=os.path.join(MESH_DIR, f"{obj.name}.obj"),
            export_selected_objects=True, forward_axis="Y", up_axis="Z",
            export_materials=False, export_uv=True, export_normals=True,
            export_triangulated_mesh=True, apply_modifiers=True)
        print("exported", obj.name, len(obj.data.polygons), "faces")


def layout_for_preview(desk, box_parts, wrap):
    """Place the props roughly like the real scene, only for looking at them in Blender."""
    for o in box_parts:
        o.location = (0.0, 0.25, DESK_H)
    for o in wrap:
        o.location = (0.0, -0.12, DESK_H)


def main():
    reset_scene()
    desk = build_desk()
    cbox = build_cardboard_box()
    wrap = build_wrapped_box()
    export(desk + cbox + wrap)  # exported at their own origins (before the preview layout)
    layout_for_preview(desk, cbox, wrap)
    bpy.ops.object.light_add(type="SUN", location=(1, -1, 3))
    bpy.context.active_object.data.energy = 3.0


main()
