"""Build the bubble-wrapped box (visual mesh + texture) from the real recordings, without Blender.

The real object is a cardboard box (23 x 23 x 27 cm) loosely wrapped in bubble wrap: a soft box with
rounded edges whose upper third bulges out (the wrap overhangs), a crease where the layers overlap, and a
crumpled top. So:
  * mesh: a rounded box (superquadric) with that profile, smooth bulges + fine wrinkles, flat bottom,
    origin at the centre of the bottom face; per-vertex normals so it shades as one soft surface;
  * texture: an atlas [+x | +y | -x | -y | top] made from what the robot's own cameras recorded, so the
    sim images have the real bubble statistics at the scale each camera sees them:
      -x (the face toward the robot) and the top: rectified from a real head-camera frame;
      +-y (the faces the palms squeeze, seen close up by the wrist cameras): real wrist-camera close-ups;
    with the recordings' own shading partly flattened (the sim's lights shade the object again).
Supersedes the wrapped box that blender_build_lab.py writes (same file names, so scene_lab.xml is unchanged).
Needs the reference frames from tools/extract_real_refs.py (calib/real_refs/); the generated mesh and
texture are committed, so this only has to run when you change the object.

    python 03_real2sim/assets/lab/make_wrapped_box.py
"""

import os

import cv2
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
REFS = os.path.join(HERE, "..", "..", "calib", "real_refs")
HEAD_FRAME = "ep000_f0000_cam_head.png"
WRIST_FRAMES = ("ep000_f0000_cam_left_wrist.png", "ep023_f0000_cam_left_wrist.png")
MESH = os.path.join(HERE, "mesh", "wrapped_box.obj")
TEX = os.path.join(HERE, "tex", "bubble_wrap.png")

HALF = np.array([0.104, 0.104, 0.133])   # with the overhang and lumps: 23-24 x 23-24 x 27 cm
ROUND = 7.0                              # superquadric exponent: higher = sharper edges
N = 30                                   # grid points per face edge
TILE = 512
# Corners in the head frame (TL, TR, BR, BL as seen in the image): the face toward the robot, and the top.
FRONT_QUAD = [(250, 133), (505, 133), (480, 343), (268, 343)]
TOP_QUAD = [(275, 28), (485, 28), (505, 133), (250, 133)]
WRIST_PATCH = (0, 250, 320, 480)         # x0, y0, x1, y1: bubble wrap filling the wrist image
# faces: (axis, sign, u axis, v axis, atlas tile)
FACES = [(0, 1, 1, 2, 0), (1, 1, 0, 2, 1), (0, -1, 1, 2, 2), (1, -1, 0, 2, 3), (2, 1, 0, 1, 4), (2, -1, 0, 1, 0)]


def bumps(q, rng_seed=5):
    """Smooth displacement [m] as a function of the normalised position q (N,3): big soft bulges, the
    pinched band, small wrinkles. The same function on every face, so the faces meet without steps."""
    rng = np.random.default_rng(rng_seed)
    d = np.zeros(len(q))
    for amp, freq, n in ((0.0050, 2.6, 5), (0.0030, 6.0, 6), (0.0014, 15.0, 8)):
        for _ in range(n):
            k = rng.normal(size=3)
            k *= freq / np.linalg.norm(k)
            d += amp / np.sqrt(n) * 1.6 * np.sin(q @ k + rng.uniform(0, 2 * np.pi))
    d += 0.008 * np.clip(q[:, 2], 0, 1) ** 6 * (1 - np.max(np.abs(q[:, :2]), axis=1) ** 2)  # puffed top
    return d


def profile(zn):
    """Horizontal scale along the height (zn in -1..1): slim body, crease, overhanging upper third."""
    upper = 1 / (1 + np.exp(-(zn - 0.22) / 0.07))
    return 0.96 + 0.07 * upper - 0.025 * np.exp(-((zn - 0.16) / 0.07) ** 2)


def build_mesh():
    verts, uvs, faces = [], [], []
    lin = np.linspace(-1, 1, N)
    uu, vv = np.meshgrid(lin, lin, indexing="ij")
    for axis, sign, ua, va, tile in FACES:
        p = np.zeros((N, N, 3))
        p[..., axis] = sign
        p[..., ua], p[..., va] = uu, vv
        p = p.reshape(-1, 3)
        q = p / (np.abs(p) ** ROUND).sum(1, keepdims=True) ** (1 / ROUND)      # cube -> rounded cube
        normal = np.sign(q) * np.abs(q) ** (ROUND - 1) / HALF
        normal /= np.linalg.norm(normal, axis=1, keepdims=True)
        x = q * HALF + normal * bumps(q)[:, None]
        x[:, :2] *= profile(q[:, 2])[:, None]
        x[:, 2] = np.maximum(x[:, 2] + HALF[2], 0.0)                           # origin = bottom face centre
        if axis == 2 and sign < 0:
            x[:, 2] = 0.0
        # Atlas UV: u runs left->right as seen from outside, v up; a 3 % margin keeps tiles from bleeding.
        u = (uu.reshape(-1) + 1) / 2
        v = (vv.reshape(-1) + 1) / 2
        flip = (axis == 0 and sign < 0) or (axis == 1 and sign > 0)
        u = 1 - u if flip else u
        u = (tile + 0.03 + 0.94 * u) / 5
        v = 0.03 + 0.94 * v
        base = len(verts)
        verts += x.tolist()
        uvs += np.stack([u, v], 1).tolist()
        # Winding: counter-clockwise seen from outside.
        ccw = np.cross(np.eye(3)[ua], np.eye(3)[va])[axis] * sign > 0
        for i in range(N - 1):
            for j in range(N - 1):
                a, b, c, d = base + i * N + j, base + (i + 1) * N + j, base + (i + 1) * N + j + 1, base + i * N + j + 1
                faces += [(a, b, c), (a, c, d)] if ccw else [(a, c, b), (a, d, c)]
    verts = np.array(verts)
    faces = np.array(faces)
    # Smooth normals across the face borders: average face normals over vertices at the same position.
    fn = np.cross(verts[faces[:, 1]] - verts[faces[:, 0]], verts[faces[:, 2]] - verts[faces[:, 0]])
    key = {}
    ids = np.array([key.setdefault(tuple(np.round(v, 5)), len(key)) for v in verts])
    acc = np.zeros((len(key), 3))
    for k in range(3):
        np.add.at(acc, ids[faces[:, k]], fn)
    vn = acc[ids]
    vn /= np.maximum(np.linalg.norm(vn, axis=1, keepdims=True), 1e-12)
    with open(MESH, "w") as f:
        f.write("# bubble-wrapped box (03_real2sim/assets/lab/make_wrapped_box.py)\n")
        for v in verts:
            f.write("v %.5f %.5f %.5f\n" % tuple(v))
        for t in uvs:
            f.write("vt %.5f %.5f\n" % tuple(t))
        for n in vn:
            f.write("vn %.4f %.4f %.4f\n" % tuple(n))
        for a, b, c in faces + 1:
            f.write(f"f {a}/{a}/{a} {b}/{b}/{b} {c}/{c}/{c}\n")
    return verts


def flatten_light(img, keep=0.5):
    """Remove part of the recording's own shading (the sim lights the object again), keep the bubbles."""
    img = img.astype(np.float32)
    low = cv2.GaussianBlur(img, (0, 0), img.shape[1] / 5)
    return img / np.maximum(low, 1) * (keep * low + (1 - keep) * low.mean((0, 1)))


def rectify(img, quad):
    M = cv2.getPerspectiveTransform(np.float32(quad), np.float32([[0, 0], [TILE, 0], [TILE, TILE], [0, TILE]]))
    return cv2.warpPerspective(img, M, (TILE, TILE), flags=cv2.INTER_CUBIC, borderMode=cv2.BORDER_REPLICATE)


def build_texture():
    head = cv2.imread(os.path.join(REFS, HEAD_FRAME))
    front = flatten_light(rectify(head, FRONT_QUAD))
    top = flatten_light(rectify(head, TOP_QUAD))[::-1]     # OBJ v runs up; the image's far edge is +x
    top *= 1.06 * front.mean() / top.mean()
    sides = []
    x0, y0, x1, y1 = WRIST_PATCH
    for name in WRIST_FRAMES:
        patch = cv2.imread(os.path.join(REFS, name))[y0:y1, x0:x1]
        side = flatten_light(cv2.resize(patch, (TILE, TILE), interpolation=cv2.INTER_CUBIC), keep=0.3)
        side = side + 0.5 * (side - cv2.GaussianBlur(side, (0, 0), 5))   # crisper bubbles
        sides.append(side * front.mean() / side.mean())
    # The head frame gives the right large-scale look but only ~1 px/mm; add the bubbles' fine detail
    # (high-pass of a wrist close-up, which is at the true scale) so the faces stay crisp close up.
    detail = sides[0] - cv2.GaussianBlur(sides[0], (0, 0), 5)
    front, top = [cv2.GaussianBlur(t, (0, 0), 1.5) + 1.5 * d for t, d in ((front, detail), (top, detail[::-1]))]
    atlas = np.hstack([front[:, ::-1], sides[0], front, sides[1], top])   # +x, +y, -x, -y, top
    cv2.imwrite(TEX, np.clip(atlas, 0, 255).astype(np.uint8))


if __name__ == "__main__":
    v = build_mesh()
    build_texture()
    print(f"wrote {MESH} ({len(v)} vertices, extents {np.round(np.ptp(v, axis=0), 3)}) and {TEX}")
