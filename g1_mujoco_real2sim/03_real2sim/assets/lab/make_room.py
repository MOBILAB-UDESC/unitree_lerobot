"""Build the lab room around the desk (real2sim of what the cameras see past the desk).

Modelled from the lab photos in ref/room/ (6-12.jpg), at real-world scale from the known sizes of the
Bambu Lab printers on the workbench (H2D 492 x 514 x 626 mm, A1 ~385 x 410 mm footprint):
  ahead (+x): past a ~1 m walkway, the workbench (wood top, blue frame, drawers + shelves) with 2x H2D
              (+ AMS 2 Pro) and an A1, a beige cabinet with two grey foam-mat cubes stacked on it, white
              wall, a plant on the left;
  right (-y): black desk with a monitor, beige cabinets and the exit door on the right wall;
  left (+y):  the long room (office with blue blinds at the far end);
  above:      coffered dark-grey concrete ceiling with fluorescent strips.
Front faces carry perspective-rectified crops of photo 11 (taken square-on to the workbench).

Writes tex/room_*.png, mesh/room_*.obj and ../../scene_room.xml (included by scene_lab.xml).

    python 03_real2sim/assets/lab/make_room.py
"""

import os

import cv2
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
REF = os.path.join(HERE, "ref", "room")
TEX = os.path.join(HERE, "tex")
MESH = os.path.join(HERE, "mesh")
XML = os.path.join(HERE, "..", "..", "scene_room.xml")

# Layout (world = robot frame: pelvis at the origin, +x forward, +y the robot's left).
# Where things stand comes from the recordings themselves: in the left wrist camera's first frames the
# foam stack on the cabinet is seen corner-on. With the camera pose known from the recorded joints, the
# bearings of its three vertical edges (-14, -21, -35 deg) and its apparent height (2 tiles = 240 px at
# f = 385 px) put its front-left corner ~1.75 m ahead and ~0.7 m to the right of the left wrist, i.e. at
# about (1.95, -0.55). The head camera's top rows (bench frame + drawers, the bench's right end ~0.5 m
# right of centre) and the walkway between desk and bench in photo 9 agree with that distance.
# Heights from photo 10: the cabinet is 1.32x the bench height; H2D (626 mm) vs bench in photo 11.
WALL_X = 2.45
BENCH_DEPTH = 0.70
BENCH_FRONT = WALL_X - 0.03 - BENCH_DEPTH   # front edge of the workbench top
BENCH_TOP_Z = 0.78
BENCH_Y = (1.40, -0.50)       # left end, right end (1.9 m long: 996 px of photo 11 at the H2D's scale)
CABINET_Y = (-0.53, -1.36)    # 0.83 m wide: 1.66x the foam tile (photo 10)
CABINET_DEPTH, CABINET_H = 0.50, 1.03
FOAM_CUBE = 0.55              # EVA floor tiles clipped into a cube; two cubes were stacked during recording
FOAM_STACK = 2
# Optional rigid correction of the whole room (03_real2sim/tools/fit_room.py prints candidates).
ROOM_SHIFT = (0.0, 0.0)
ROOM_YAW_DEG = 0.0
CABINET_DZ = 0.0
RIGHT_WALL_Y, LEFT_WALL_Y, BACK_WALL_X, CEILING_Z = -3.2, 7.5, -4.5, 3.0


def rect(img, quad, w, h):
    M = cv2.getPerspectiveTransform(np.float32(quad), np.float32([[0, 0], [w, 0], [w, h], [0, h]]))
    return cv2.warpPerspective(img, M, (w, h), flags=cv2.INTER_AREA)


def atlas(name, front, side_bgr, top_bgr):
    """Texture = [front crop | side colour | top colour]; returns the UV boxes (u0, v0, u1, v1)."""
    h, w = front.shape[:2]
    pad = 32
    img = np.zeros((h, w + 2 * pad, 3), np.uint8)
    img[:, :w] = front
    img[:, w:w + pad] = side_bgr
    img[:, w + pad:] = top_bgr
    cv2.imwrite(os.path.join(TEX, f"room_{name}.png"), img)
    W = w + 2 * pad
    mid = lambda u: (u / W + 0.25 / W, 0.25, u / W + 0.75 / W, 0.75)  # noqa: E731 - a tiny patch in a column
    return (0.0, 0.0, w / W, 1.0), mid(w + pad / 2), mid(w + 1.5 * pad)


def write_box_obj(name, size, front_uv, side_uv, top_uv):
    """Box centred at the origin; its -x face (toward the robot) shows the front crop upright, image left
    on +y (the robot's left), so the crop reads as in the photo. Per-face vertices and normals."""
    sx, sy, sz = np.asarray(size) / 2
    faces = {  # normal: (4 corners CCW seen from outside, uv box)
        (-1, 0, 0): ([(-sx, sy, -sz), (-sx, -sy, -sz), (-sx, -sy, sz), (-sx, sy, sz)], front_uv),
        (1, 0, 0): ([(sx, -sy, -sz), (sx, sy, -sz), (sx, sy, sz), (sx, -sy, sz)], side_uv),
        (0, 1, 0): ([(sx, sy, -sz), (-sx, sy, -sz), (-sx, sy, sz), (sx, sy, sz)], side_uv),
        (0, -1, 0): ([(-sx, -sy, -sz), (sx, -sy, -sz), (sx, -sy, sz), (-sx, -sy, sz)], side_uv),
        (0, 0, 1): ([(-sx, -sy, sz), (sx, -sy, sz), (sx, sy, sz), (-sx, sy, sz)], top_uv),
        (0, 0, -1): ([(-sx, sy, -sz), (sx, sy, -sz), (sx, -sy, -sz), (-sx, -sy, -sz)], side_uv),
    }
    lines, n = [f"# {name} (03_real2sim/assets/lab/make_room.py)"], 0
    for normal, (corners, (u0, v0, u1, v1)) in faces.items():
        for c in corners:
            lines.append("v %.4f %.4f %.4f" % c)
        for uv in ((u0, v0), (u1, v0), (u1, v1), (u0, v1)):
            lines.append("vt %.5f %.5f" % uv)
        lines.append("vn %d %d %d" % normal)
        k = n // 4 + 1
        a, b, c_, d = n + 1, n + 2, n + 3, n + 4
        lines.append(f"f {a}/{a}/{k} {b}/{b}/{k} {c_}/{c_}/{k}")
        lines.append(f"f {a}/{a}/{k} {c_}/{c_}/{k} {d}/{d}/{k}")
        n += 4
    with open(os.path.join(MESH, f"room_{name}.obj"), "w") as f:
        f.write("\n".join(lines) + "\n")


def ceiling_texture():
    """Coffered concrete ceiling, one 1.2 m bay per texture repeat: dark recess, lighter ribs."""
    s = 256
    img = np.full((s, s, 3), (62, 64, 66), np.uint8)
    rib = 22
    img[:rib], img[:, :rib] = (118, 120, 122), (118, 120, 122)
    img[rib:rib + 10], img[:, rib:rib + 10] = (88, 90, 92), (88, 90, 92)   # rib bevel
    cv2.imwrite(os.path.join(TEX, "room_ceiling.png"), img)


def main():
    p11 = cv2.imread(os.path.join(REF, "11.jpg"))
    p9 = cv2.imread(os.path.join(REF, "9.jpg"))
    blue, white, black, beige = (150, 95, 30), (225, 226, 226), (30, 30, 32), (190, 214, 224)

    # (name, crop quad in photo 11, crop px, side BGR, top BGR, box size [x depth, y width, z height])
    boxes = {
        "bench_frame": ([(297, 560), (1213, 560), (1163, 887), (350, 887)], (900, 400), blue, blue,
                        (BENCH_DEPTH - 0.06, BENCH_Y[0] - BENCH_Y[1] - 0.10, BENCH_TOP_Z - 0.04)),
        "bench_top": ([(252, 520), (1248, 520), (1248, 560), (252, 560)], (900, 40), (40, 110, 190),
                      (55, 125, 200), (BENCH_DEPTH, BENCH_Y[0] - BENCH_Y[1], 0.04)),
        "h2d_left": ([(385, 188), (632, 196), (630, 498), (412, 498)], (250, 320), white, black,
                     (0.514, 0.492, 0.626)),
        "h2d_right": ([(720, 192), (962, 192), (962, 505), (735, 505)], (250, 320), white, black,
                      (0.514, 0.492, 0.626)),
        "ams_left": ([(425, 100), (635, 100), (635, 192), (425, 192)], (210, 95), (70, 70, 72), (95, 95, 98),
                     (0.28, 0.37, 0.19)),
        "ams_right": ([(742, 100), (918, 100), (918, 192), (742, 192)], (180, 95), (70, 70, 72),
                      (95, 95, 98), (0.28, 0.37, 0.19)),
        "a1": ([(985, 190), (1145, 190), (1145, 497), (985, 497)], (160, 300), white, white, (0.41, 0.33, 0.62)),
        "cabinet": ([(1212, 428), (1588, 422), (1500, 800), (1207, 800)], (380, 400), beige, beige,
                    (CABINET_DEPTH, CABINET_Y[0] - CABINET_Y[1], CABINET_H + CABINET_DZ)),
        "foam_cube": ([(1305, 185), (1535, 190), (1490, 398), (1290, 410)], (300, 300), (95, 96, 98),
                      (95, 96, 98), (FOAM_CUBE,) * 3),
    }
    for name, (quad, (w, h), side, top, size) in boxes.items():
        front = rect(p11, quad, w, h)
        if name == "foam_cube":  # same tile pattern on every face
            side = top = None
            uv = (0.0, 0.0, 1.0, 1.0)
            cv2.imwrite(os.path.join(TEX, f"room_{name}.png"), front)
            write_box_obj(name, size, uv, uv, uv)
            continue
        write_box_obj(name, size, *atlas(name, front, side, top))
    # Far end of the long room on the robot's left: blinds + office desks (photo 9), a flat billboard.
    office = rect(p9, [(290, 500), (830, 500), (830, 870), (290, 870)], 540, 370)
    cv2.imwrite(os.path.join(TEX, "room_office.png"), office)
    ceiling_texture()

    # ---- MJCF ----
    bf, bd = BENCH_FRONT, BENCH_DEPTH
    bench_cy = (BENCH_Y[0] + BENCH_Y[1]) / 2
    on_bench = lambda frac: BENCH_Y[0] - frac * (BENCH_Y[0] - BENCH_Y[1])  # noqa: E731 (photo 11 x fraction)
    pf = bf + 0.08  # printers stand back from the bench edge
    tz = BENCH_TOP_Z
    cab_cx, cab_cy = WALL_X - CABINET_DEPTH / 2, sum(CABINET_Y) / 2
    textured = [  # (mesh, pos)
        ("bench_frame", (bf + 0.03 + (bd - 0.06) / 2, bench_cy, (tz - 0.04) / 2)),
        ("bench_top", (bf + bd / 2, bench_cy, tz - 0.02)),
        ("h2d_left", (pf + 0.257, on_bench(0.261), tz + 0.313)),
        ("ams_left", (pf + 0.20, on_bench(0.261), tz + 0.626 + 0.095)),
        ("h2d_right", (pf + 0.257, on_bench(0.592), tz + 0.313)),
        ("ams_right", (pf + 0.20, on_bench(0.592), tz + 0.626 + 0.095)),
        ("a1", (pf + 0.205, on_bench(0.82), tz + 0.31)),
        ("cabinet", (cab_cx, cab_cy, (CABINET_H + CABINET_DZ) / 2)),
    ]
    cab_front = WALL_X - CABINET_DEPTH
    cube_xy = (cab_front + 0.02 + FOAM_CUBE / 2, CABINET_Y[0] - 0.02 - FOAM_CUBE / 2)
    cubes = [(f"foam_cube{k}", (*cube_xy, CABINET_H + CABINET_DZ + FOAM_CUBE * (k + 0.5))) for k in range(FOAM_STACK)]
    assets = ["    <texture name=\"room_ceiling\" type=\"2d\" file=\"room_ceiling.png\"/>",
              "    <material name=\"room_ceiling\" texture=\"room_ceiling\" texrepeat=\"0.8333 0.8333\" "
              "texuniform=\"true\" specular=\"0\"/>",
              "    <texture name=\"room_office\" type=\"2d\" file=\"room_office.png\"/>",
              "    <material name=\"room_office\" texture=\"room_office\" specular=\"0\"/>",
              "    <material name=\"room_wall\" rgba=\"0.90 0.90 0.89 1\" specular=\"0.05\"/>",
              "    <material name=\"room_black\" rgba=\"0.07 0.07 0.08 1\" specular=\"0.3\"/>",
              "    <material name=\"room_kraft\" rgba=\"0.72 0.55 0.37 1\" specular=\"0.05\"/>",
              "    <material name=\"room_beige\" rgba=\"0.88 0.84 0.74 1\" specular=\"0.1\"/>",
              "    <material name=\"room_door\" rgba=\"0.62 0.35 0.18 1\" specular=\"0.2\"/>",
              "    <material name=\"room_red\" rgba=\"0.75 0.12 0.10 1\"/>",
              "    <material name=\"room_chair\" rgba=\"0.10 0.22 0.65 1\" specular=\"0.1\"/>",
              "    <material name=\"room_leaf\" rgba=\"0.20 0.38 0.16 1\" specular=\"0.1\"/>",
              "    <material name=\"room_lamp\" rgba=\"1 1 1 1\" emission=\"1\"/>",
              "    <material name=\"room_board\" rgba=\"0.96 0.96 0.97 1\" specular=\"0.4\"/>",
              "    <material name=\"room_metal\" rgba=\"0.62 0.63 0.65 1\" specular=\"0.5\"/>"]
    body = []
    for name, _ in textured:
        assets.append(f"    <texture name=\"room_{name}\" type=\"2d\" file=\"room_{name}.png\"/>")
        assets.append(f"    <material name=\"room_{name}\" texture=\"room_{name}\" specular=\"0.15\"/>")
        assets.append(f"    <mesh name=\"room_{name}\" file=\"room_{name}.obj\"/>")
    for name, pos in textured:
        body.append(f"    <geom name=\"room_{name}\" class=\"visual\" type=\"mesh\" mesh=\"room_{name}\" "
                    f"material=\"room_{name}\" pos=\"{pos[0]:.3f} {pos[1]:.3f} {pos[2]:.3f}\"/>")
    assets += ["    <texture name=\"room_foam_cube\" type=\"2d\" file=\"room_foam_cube.png\"/>",
               "    <material name=\"room_foam_cube\" texture=\"room_foam_cube\" specular=\"0.05\"/>",
               "    <mesh name=\"room_foam_cube\" file=\"room_foam_cube.obj\"/>"]
    for name, pos in cubes:
        body.append(f"    <geom name=\"room_{name}\" class=\"visual\" type=\"mesh\" mesh=\"room_foam_cube\" "
                    f"material=\"room_foam_cube\" pos=\"{pos[0]:.3f} {pos[1]:.3f} {pos[2]:.3f}\"/>")

    def box(name, mat, pos, size):  # size = full extents
        body.append(f"    <geom name=\"room_{name}\" class=\"visual\" type=\"box\" material=\"room_{mat}\" "
                    f"pos=\"{pos[0]:.3f} {pos[1]:.3f} {pos[2]:.3f}\" "
                    f"size=\"{size[0] / 2:.3f} {size[1] / 2:.3f} {size[2] / 2:.3f}\"/>")

    # Bench-top clutter (photo 11): filament box, dryer, spool, small cardboard box.
    box("filament_box", "black", (pf + 0.16, on_bench(0.12), tz + 0.06), (0.32, 0.17, 0.12))
    box("dryer", "black", (pf + 0.09, on_bench(0.43), tz + 0.125), (0.18, 0.10, 0.25))
    body.append(f"    <geom name=\"room_spool\" class=\"visual\" type=\"cylinder\" material=\"room_black\" "
                f"pos=\"{bf + 0.15:.3f} {on_bench(0.73):.3f} {tz + 0.035:.3f}\" size=\"0.10 0.035\"/>")
    box("small_box", "kraft", (bf + 0.20, on_bench(0.92), tz + 0.04), (0.22, 0.30, 0.08))
    # On the cabinet, right of the foam stack (wrist recordings): a small cardboard box; blue tape on the door.
    box("cabinet_box", "kraft", (cab_front + 0.14, CABINET_Y[1] + 0.13, CABINET_H + CABINET_DZ + 0.07),
        (0.2, 0.2, 0.14))
    # Walls, ceiling, lamps.
    wy0, wy1 = RIGHT_WALL_Y, LEFT_WALL_Y
    box("wall_front", "wall", (WALL_X + 0.05, (wy0 + wy1) / 2, CEILING_Z / 2), (0.1, wy1 - wy0, CEILING_Z))
    box("wall_right", "wall", ((BACK_WALL_X + WALL_X) / 2, wy0 - 0.05, CEILING_Z / 2),
        (WALL_X - BACK_WALL_X, 0.1, CEILING_Z))
    box("wall_left", "wall", ((BACK_WALL_X + WALL_X) / 2, wy1 + 0.05, CEILING_Z / 2),
        (WALL_X - BACK_WALL_X, 0.1, CEILING_Z))
    box("wall_back", "wall", (BACK_WALL_X - 0.05, (wy0 + wy1) / 2, CEILING_Z / 2), (0.1, wy1 - wy0, CEILING_Z))
    body.append(f"    <geom name=\"room_office\" class=\"visual\" type=\"box\" material=\"room_office\" "
                f"pos=\"-1.2 {wy1 - 0.01:.3f} 1.25\" size=\"2.6 0.005 1.8\" euler=\"0 0 180\"/>")
    box("ceiling", "ceiling", ((BACK_WALL_X + WALL_X) / 2, (wy0 + wy1) / 2, CEILING_Z + 0.05),
        (WALL_X - BACK_WALL_X, wy1 - wy0, 0.1))
    for i, x in enumerate((-2.6, -0.2)):
        for j, y in enumerate((-1.6, 1.0, 3.6)):
            box(f"lamp_{i}{j}", "lamp", (x + 1.0, y, CEILING_Z - 0.03), (1.2, 0.16, 0.04))
    # Whiteboard and plant left of the workbench.
    plant_y = BENCH_Y[0] + 0.45
    box("whiteboard", "board", (WALL_X - 0.01, BENCH_Y[0] + 1.1, 1.45), (0.02, 1.2, 0.9))
    body.append("    <geom name=\"room_pot\" class=\"visual\" type=\"cylinder\" material=\"room_black\" "
                f"pos=\"{WALL_X - 0.35:.3f} {plant_y:.3f} 0.15\" size=\"0.17 0.15\"/>")
    body.append("    <geom name=\"room_trunk\" class=\"visual\" type=\"cylinder\" material=\"room_kraft\" "
                f"pos=\"{WALL_X - 0.35:.3f} {plant_y:.3f} 0.75\" size=\"0.025 0.45\"/>")
    for k, (dy, dz, r) in enumerate(((0, 1.35, 0.32), (0.12, 1.0, 0.25), (-0.1, 1.65, 0.22))):
        body.append(f"    <geom name=\"room_leaves{k}\" class=\"visual\" type=\"ellipsoid\" material=\"room_leaf\" "
                    f"pos=\"{WALL_X - 0.35:.3f} {plant_y + dy:.3f} {dz:.3f}\" size=\"{r:.2f} {r:.2f} {r * 1.2:.2f}\"/>")
    # Right side (photos 6-7): black desk with monitor and a blue chair, beige cabinets, exit door.
    box("desk2_top", "black", (0.1, -1.55, 0.74), (1.6, 0.8, 0.03))
    for dx in (-0.65, 0.85):
        box(f"desk2_leg{dx:+.0f}", "black", (dx, -1.55, 0.36), (0.06, 0.7, 0.72))
    box("monitor", "black", (0.2, -1.70, 1.07), (0.04, 0.62, 0.38))
    box("monitor_arm", "black", (0.25, -1.82, 0.88), (0.04, 0.04, 0.25))
    box("chair_seat", "chair", (0.15, -1.0, 0.48), (0.48, 0.48, 0.08))
    box("chair_back", "chair", (0.15, -0.78, 0.80), (0.42, 0.06, 0.5))
    for k, x in enumerate(np.arange(-3.6, 1.4, 0.9)):
        box(f"side_cabinet{k}", "beige", (x, RIGHT_WALL_Y + 0.25, 0.45), (0.86, 0.5, 0.9))
    box("door", "door", (1.0, RIGHT_WALL_Y + 0.02, 1.05), (0.9, 0.04, 2.1))
    box("door_sign", "red", (1.0, RIGHT_WALL_Y + 0.03, 2.25), (0.3, 0.02, 0.1))

    with open(XML, "w") as f:
        f.write("<!-- Lab room around the desk. Generated by assets/lab/make_room.py - edit that, not this. -->\n")
        f.write("<mujoco>\n  <asset>\n" + "\n".join(assets) + "\n  </asset>\n")
        # Everything turns about the workbench top's centre, then shifts (ROOM_* from fit_room.py).
        px, py = bf + bd / 2, bench_cy
        h = np.radians(ROOM_YAW_DEG) / 2
        f.write("  <worldbody>\n")
        f.write(f"   <frame pos=\"{px + ROOM_SHIFT[0]:.4f} {py + ROOM_SHIFT[1]:.4f} 0\" "
                f"quat=\"{np.cos(h):.6f} 0 0 {np.sin(h):.6f}\">\n")
        f.write(f"   <frame pos=\"{-px:.4f} {-py:.4f} 0\">\n")
        f.write("\n".join(body) + "\n   </frame>\n   </frame>\n  </worldbody>\n</mujoco>\n")
    print(f"wrote {XML} ({len(body)} geoms)")


if __name__ == "__main__":
    main()
