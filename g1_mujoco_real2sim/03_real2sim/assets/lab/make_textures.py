"""Build the lab textures from the reference photos in ref/ (real2sim).

Colours are sampled from clean patches of the photos; grain, flutes and bubbles are synthesised so
the textures tile without the background that a raw crop would drag along. Output: tex/*.png.
Run with any python that has opencv + numpy:  python make_textures.py
"""

import os

import cv2
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
REF = os.path.join(HERE, "ref")
OUT = os.path.join(HERE, "tex")
rng = np.random.default_rng(7)


def patch(photo, x0, y0, x1, y1):
    return cv2.imread(os.path.join(REF, f"{photo}.jpg"))[y0:y1, x0:x1].astype(np.float32)


def stats(p):
    return p.reshape(-1, 3).mean(0), p.reshape(-1, 3).std(0)


def noise(size, scale, octaves=4):
    """Smooth value noise in [-1, 1]."""
    acc = np.zeros((size, size), np.float32)
    amp, total = 1.0, 0.0
    for o in range(octaves):
        n = max(2, int(size / scale * 2 ** o))
        layer = cv2.resize(rng.standard_normal((n, n)).astype(np.float32), (size, size), interpolation=cv2.INTER_CUBIC)
        acc += amp * layer
        total += amp
        amp *= 0.5
    acc /= total
    return acc / (np.abs(acc).max() + 1e-6)


def tileable(img):
    """Mirror-tile at half resolution: seamless at any repeat, no blend seams."""
    h, w = img.shape[:2]
    half = cv2.resize(img, (w // 2, h // 2), interpolation=cv2.INTER_AREA)
    top = np.hstack([half, half[:, ::-1]])
    return np.vstack([top, top[::-1]])


def save(name, img):
    cv2.imwrite(os.path.join(OUT, name), np.clip(img, 0, 255).astype(np.uint8))
    print("wrote", name)


def cardboard(name, mean, flute_axis=1, size=512):
    """Kraft board: base colour, fibrous low-freq mottling, faint corrugation flutes."""
    base = np.ones((size, size, 3), np.float32) * mean
    mottle = noise(size, 64)[..., None] * 9 + noise(size, 8, 2)[..., None] * 4
    t = np.arange(size, dtype=np.float32)
    flutes = (np.sin(t / size * 2 * np.pi * 40) * 2.5)
    flutes = flutes[None, :, None] if flute_axis == 1 else flutes[:, None, None]
    fibres = rng.standard_normal((size, size, 1)).astype(np.float32) * 3
    return tileable(base + mottle + flutes + fibres)


def main():
    os.makedirs(OUT, exist_ok=True)

    out_mean, _ = stats(patch(1, 190, 450, 500, 570))   # outer wall, lit side
    in_mean, _ = stats(patch(2, 220, 800, 400, 950))    # inside floor (lighter, pinkish)
    save("cardboard_out.png", cardboard("out", out_mean))
    save("cardboard_in.png", cardboard("in", in_mean, flute_axis=0))

    # Desk laminate: blue with very fine speckle (photo 2 top view, photo 3 oblique).
    top_mean, _ = stats(np.concatenate([patch(2, 120, 620, 170, 1100).reshape(-1, 3),
                                        patch(3, 250, 860, 420, 990).reshape(-1, 3)]))
    size = 512
    desk = np.ones((size, size, 3), np.float32) * top_mean
    desk += noise(size, 128)[..., None] * 6 + rng.standard_normal((size, size, 1)).astype(np.float32) * 2.5
    save("desk_top.png", tileable(desk))

    panel_mean, _ = stats(patch(3, 420, 1012, 860, 1060))
    panel = np.ones((size, size, 3), np.float32) * panel_mean
    panel += noise(size, 96)[..., None] * 4 + rng.standard_normal((size, size, 1)).astype(np.float32) * 1.5
    save("desk_panel.png", tileable(panel))

    # Bubble wrap: real photo detail, made tileable, plus a synthetic bubble lattice for the
    # characteristic dotted highlights. Light, slightly cool white.
    real = patch(3, 500, 600, 700, 740)   # upper part of the wrapped box, no desk
    real = cv2.resize(real, (size, size), interpolation=cv2.INTER_CUBIC)
    real = tileable(real)
    lattice = np.zeros((size, size), np.float32)
    pitch = 18
    for row, y in enumerate(range(0, size + pitch, pitch)):
        for x in range(-pitch, size + pitch, pitch):
            cx = x + (pitch // 2 if row % 2 else 0)
            cv2.circle(lattice, (cx % size, y % size), 6, 1.0, -1, lineType=cv2.LINE_AA)
            cv2.circle(lattice, ((cx - 2) % size, (y - 2) % size), 2, 2.0, -1, lineType=cv2.LINE_AA)
    lattice = cv2.GaussianBlur(lattice, (0, 0), 1.2)
    bubble = 0.75 * real + 0.25 * np.full_like(real, 225) + lattice[..., None] * 18
    save("bubble_wrap.png", np.clip(bubble, 0, 255))

    # Desk legs / frame: near-black satin metal, and glove knit (black, slight texture).
    legs = np.full((128, 128, 3), 28, np.float32) + rng.standard_normal((128, 128, 1)).astype(np.float32) * 2
    save("desk_leg.png", legs)
    knit = np.full((256, 256, 3), 24, np.float32)
    yy, xx = np.mgrid[0:256, 0:256].astype(np.float32)
    knit += (np.sin(xx / 3.0) * np.sin(yy / 5.0) * 6)[..., None] + rng.standard_normal((256, 256, 1)).astype(np.float32) * 3
    save("glove_knit.png", knit)


if __name__ == "__main__":
    main()
