import math as m
import time

import numpy as np

__author__ = "Mahamad (aGsx / @aGsx-bit)"
__copyright__ = "Copyright (c) 2026 Mahamad (aGsx / @aGsx-bit)"
__license__ = "Attribution Open Source (see LICENSE)"

SIZE = 6
FPS = 30
SS = 1
ASPECT = 1.46
MAXROWS = 96
CW_GLYPH = 4
CH_GLYPH = 7

T_BODY0 = 0.90
T_FACE0 = 2.35
T_ARMS0 = 3.85
T_SCARF0 = 5.05
T_HAT0 = 6.75
T_HEART0 = 8.20
T_FINAL = 9.50

RAMP = " .'`:-=+*#%8MW@"
NR = len(RAMP) - 1
CIDX = {c: i for i, c in enumerate(RAMP)}
SOLID_FLOOR = 10
QL = 24

_GLYPH_PATTERNS = {
    " ": ("....", "....", "....", "....", "....", "....", "...."),
    ".": ("....", "....", "....", "....", "....", ".#..", "...."),
    "'": (".#..", ".#..", "....", "....", "....", "....", "...."),
    "`": ("#...", ".#..", "....", "....", "....", "....", "...."),
    ":": ("....", ".#..", "....", "....", ".#..", "....", "...."),
    "-": ("....", "....", "....", "###.", "....", "....", "...."),
    "=": ("....", "###.", "....", "###.", "....", "....", "...."),
    "+": ("....", ".#..", "###.", ".#..", "....", "....", "...."),
    "*": ("#.#.", ".#..", "###.", ".#..", "#.#.", "....", "...."),
    "#": ("#.#.", "####", "#.#.", "####", "#.#.", "#.#.", "...."),
    "%": ("#..#", "..#.", ".#..", ".#..", "#..#", "....", "...."),
    "8": (".##.", "#..#", ".##.", "#..#", "#..#", ".##.", "...."),
    "M": ("#..#", "####", "#.##", "#..#", "#..#", "#..#", "...."),
    "W": ("#..#", "#..#", "#..#", "#.##", "####", "#..#", "...."),
    "@": (".##.", "#..#", "#.##", "#.##", "#...", ".###", "...."),
}

GLYPH_MASK = np.zeros((len(RAMP), CH_GLYPH, CW_GLYPH, 1), dtype=np.uint8)
for _i, _ch in enumerate(RAMP):
    _rows = _GLYPH_PATTERNS[_ch]
    for _r in range(CH_GLYPH):
        for _c in range(CW_GLYPH):
            GLYPH_MASK[_i, _r, _c, 0] = 1 if _rows[_r][_c] == "#" else 0
GLYPH_BG = ((1 - GLYPH_MASK) * np.array([1, 2, 3], dtype=np.uint8)).astype(np.uint8)

rng = np.random.RandomState(17)


def ss(a, b, x):
    t = np.clip((x - a) / (b - a), 0.0, 1.0)
    return t * t * (3.0 - 2.0 * t)


def ease(t):
    t = min(max(t, 0.0), 1.0)
    return t * t * (3.0 - 2.0 * t)


def ease_back(t):
    t = min(max(t, 0.0), 1.0)
    k = t * t * (3.0 - 2.0 * t)
    return min(1.06, k * (1.0 + 0.06 * m.sin(m.pi * k)))


def mix(a, b, k):
    return a + (b - a) * k


def _up2(b, n):
    g = b.shape[0]
    idx = np.arange(n) * g / n
    i0 = idx.astype(np.int64) % g
    i1 = (i0 + 1) % g
    f = idx - np.floor(idx)
    f = (f * f * (3.0 - 2.0 * f)).astype(np.float32)
    t = b[:, i0] * (1.0 - f) + b[:, i1] * f
    return t[i0, :] * (1.0 - f)[:, None] + t[i1, :] * f[:, None]


def _norm(n, k=4.0):
    return np.clip(0.5 + (n - n.mean()) / (n.std() * k + 1e-9), 0.0, 1.0).astype(np.float32)


def tex2d(n, grids):
    out = np.zeros((n, n), np.float32)
    amp = 1.0
    for g in grids:
        out += amp * _up2(rng.rand(g, g).astype(np.float32), n)
        amp *= 0.55
    return _norm(out)


def bil(tex, u, v):
    n = tex.shape[0]
    u0 = np.floor(u)
    v0 = np.floor(v)
    fu = u - u0
    fv = v - v0
    i0 = u0.astype(np.int32) % n
    j0 = v0.astype(np.int32) % n
    i1 = (i0 + 1) % n
    j1 = (j0 + 1) % n
    return (
        (tex[j0, i0] * (1.0 - fu) + tex[j0, i1] * fu) * (1.0 - fv)
        + (tex[j1, i0] * (1.0 - fu) + tex[j1, i1] * fu) * fv
    )


def sd_capsule(u, v, ax, ay, bx, by, r):
    pax, pay = u - ax, v - ay
    bax, bay = bx - ax, by - ay
    h = np.clip((pax * bax + pay * bay) / (bax * bax + bay * bay + 1e-9), 0.0, 1.0)
    return np.sqrt((pax - bax * h) ** 2 + (pay - bay * h) ** 2) - r


def sd_box(u, v, bx, by, rad=0.0):
    qx = np.abs(u) - bx + rad
    qy = np.abs(v) - by + rad
    return np.sqrt(np.maximum(qx, 0.0) ** 2 + np.maximum(qy, 0.0) ** 2) + np.minimum(np.maximum(qx, qy), 0.0) - rad


def sd_ellipse(u, v, cx, cy, rx, ry):
    du = (u - cx) / max(rx, 1e-5)
    dv = (v - cy) / max(ry, 1e-5)
    return (np.sqrt(du * du + dv * dv) - 1.0) * min(rx, ry)


def roi(cx, cy, hw, hh):
    c0 = max(0, int((cx - hw - X0_MIN) / DX_PIX))
    c1 = min(GW, int((cx + hw - X0_MIN) / DX_PIX) + 1)
    r0 = max(0, int((cy - hh - Y0_MIN) / DY_PIX))
    r1 = min(GH, int((cy + hh - Y0_MIN) / DY_PIX) + 1)
    return slice(r0, r1), slice(c0, c1)


print("Preparing textures...")
TM = tex2d(512, (6, 16, 42))
TS = tex2d(512, (5, 14, 36, 90))
TP = tex2d(512, (8, 24, 64))

NSTARS = 56
STX = rng.uniform(-0.71, 0.71, NSTARS).astype(np.float32)
STY = rng.uniform(-0.46, 0.14, NSTARS).astype(np.float32)
STP = rng.uniform(0.0, 6.28, NSTARS).astype(np.float32)
STT = rng.rand(NSTARS).astype(np.float32)

NSNOW = 95
SN_X0 = rng.uniform(-0.73, 0.73, NSNOW).astype(np.float32)
SN_Y0 = rng.uniform(-0.50, 0.50, NSNOW).astype(np.float32)
SN_SPD = rng.uniform(0.055, 0.130, NSNOW).astype(np.float32)
SN_SWAY = rng.uniform(0.010, 0.026, NSNOW).astype(np.float32)
SN_FRQ = rng.uniform(0.8, 2.2, NSNOW).astype(np.float32)
SN_PH = rng.uniform(0.0, 6.28, NSNOW).astype(np.float32)
SN_KIND = rng.rand(NSNOW).astype(np.float32)

S = {"last": 0.0, "settled_sm": None, "settled_sm_mask": None}

PINE_DEEP = np.array([0.05, 0.34, 0.22], np.float32)
PINE_MID = np.array([0.16, 0.74, 0.48], np.float32)
PINE_LIT = np.array([0.30, 0.92, 0.64], np.float32)
FROST_COOL = np.array([0.78, 0.96, 0.96], np.float32)
FROST_WHITE = np.array([1.00, 1.00, 1.00], np.float32)
TRUNK_COL = np.array([0.24, 0.16, 0.10], np.float32)
AUR_C1 = np.array([0.34, 0.72, 0.62], np.float32)
AUR_C2 = np.array([0.24, 0.56, 0.72], np.float32)
GND_DOT_COL = np.array([0.44, 0.52, 0.64], np.float32)


def ground_height(x):
    return np.full_like(x, 0.165, dtype=np.float32) + 0.006 * np.cos(x * 6.5)


def init(cw, ch, rows):
    global CW, CH, ROWS, COLS, W, H, GW, GH, X, Y, XC, YC, HSC, X0_MIN, Y0_MIN, DX_PIX, DY_PIX
    global SNOW_TEX, NEEDLE_TEX, BASE_BG, BASE_SOLID, MOON_MASK, GND_ROW0
    global XS_SKY, YS_SKY, SKY_FADE, MOON_RS, MOON_CS, MOON_COL, MOON_ALPHA, SM_RS, SM_CS
    global GR_IDX, GC_IDX, GND_WAVE_POS, GND_GLIT_SEED, GND_HIGH, PPM_HDR, Q_TO_U8
    CW = CW_GLYPH
    CH = CH_GLYPH
    ROWS = MAXROWS // 2 * 2
    COLS = int(ROWS * CH * ASPECT / CW) // 2 * 2
    W, H = COLS * CW, ROWS * CH
    GW, GH = COLS * SS, ROWS * SS
    PPM_HDR = f"P6\n{W} {H}\n255\n".encode("ascii")
    Q_TO_U8 = np.array([v * 255 // (QL - 1) for v in range(QL)], dtype=np.uint8)

    xs = (np.arange(GW, dtype=np.float32) + 0.5) * (CW / SS)
    ys = (np.arange(GH, dtype=np.float32) + 0.5) * (CH / SS)
    X = (((xs - W / 2) / H)[None, :] * np.ones((GH, 1), np.float32)).astype(np.float32)
    Y = (((ys - H / 2) / H)[:, None] * np.ones((1, GW), np.float32)).astype(np.float32)
    X0_MIN = float(X[0, 0])
    Y0_MIN = float(Y[0, 0])
    DX_PIX = float(X[0, 1] - X[0, 0])
    DY_PIX = float(Y[1, 0] - Y[0, 0])

    XC = X.reshape(ROWS, SS, COLS, SS).mean((1, 3)) if SS > 1 else X
    YC = Y.reshape(ROWS, SS, COLS, SS).mean((1, 3)) if SS > 1 else Y

    rr, cc = np.mgrid[0:ROWS, 0:COLS]
    HSC = (((rr * 73856093) ^ (cc * 19349663) ^ 0x5BD1E995) % 1009 / 1009.0).astype(np.float32)

    gy_c = ground_height(XC)
    gnd_cell = (YC >= gy_c - 0.004)
    gnd_seed = (((rr * 2654435761) ^ (cc * 2246822519)) % 997 / 997.0).astype(np.float32)
    GR_IDX, GC_IDX = np.nonzero(gnd_cell)
    GND_WAVE_POS = (XC[GR_IDX, GC_IDX] * 8.5 + YC[GR_IDX, GC_IDX] * 14.0).astype(np.float32)
    GND_GLIT_SEED = gnd_seed[GR_IDX, GC_IDX]
    GND_HIGH = GND_GLIT_SEED > 0.85

    GND_ROW0 = max(0, int((0.14 - Y0_MIN) / DY_PIX) + 1)
    XS_SKY = X[:GND_ROW0, :]
    YS_SKY = Y[:GND_ROW0, :]
    SKY_FADE = (ss(0.12, -0.06, YS_SKY) * ss(-0.50, -0.43, YS_SKY)).astype(np.float32)

    SNOW_TEX = (0.96 + 0.06 * bil(TS, X * 50.0 + 15.0, Y * 50.0 + 25.0)).astype(np.float32)
    NEEDLE_TEX = bil(TP, X * 70.0 + 12.0, Y * 70.0 + 20.0).astype(np.float32)

    px = 1.0 / GH
    bg = np.zeros((GH, GW, 3), np.float32)
    solid = np.zeros((GH, GW), np.float32)

    mx, my, mr = 0.455, -0.340, 0.064
    MOON_RS, MOON_CS = roi(mx, my, mr + 0.06, mr + 0.06)
    Xm, Ym = X[MOON_RS, MOON_CS], Y[MOON_RS, MOON_CS]
    dm = np.sqrt((Xm - mx) ** 2 + (Ym - my) ** 2)
    m_moon = np.clip((mr - dm) / (1.5 * px) + 0.5, 0.0, 1.0)
    halo = np.exp(-np.clip(dm - mr, 0.0, 1.0) * 30.0) * 0.22
    moon_tex = bil(TM, (Xm - mx) * 34.0 + 40.0, (Ym - my) * 34.0 + 90.0)
    moon_limb = 0.78 + 0.22 * ss(mr, 0.0, dm)
    moon_col = mix(
        np.array([0.74, 0.79, 0.85], np.float32),
        np.array([1.00, 0.99, 0.96], np.float32),
        ss(0.15, 0.72, moon_tex)[..., None],
    ) * moon_limb[..., None]
    MOON_ALPHA = m_moon[..., None].astype(np.float32)
    MOON_COL = (moon_col * MOON_ALPHA + halo[..., None] * np.array([0.48, 0.66, 0.96], np.float32)).astype(np.float32)
    MOON_MASK = np.zeros((ROWS, COLS), dtype=bool)
    dm_c = np.sqrt((XC - mx) ** 2 + (YC - my) ** 2)
    MOON_MASK = dm_c <= (mr + 0.012)

    gy = ground_height(X)
    m_gnd = np.clip((Y - gy) / (1.5 * px) + 0.5, 0.0, 1.0)
    g_col = np.full((GH, GW, 3), 0.004, np.float32)
    g_mask = np.zeros((GH, GW), np.float32)

    for sx_c in (-0.210, 0.210):
        d_mound = ((X - sx_c) / 0.205) ** 2 + ((Y - 0.322) / 0.050) ** 2
        m_pad = np.clip((1.0 - d_mound) * 2.5, 0.0, 1.0)
        pad_top = ss(1.0, 0.16, d_mound)
        pad_col = mix(
            np.array([0.42, 0.74, 1.00], np.float32),
            np.array([0.94, 0.99, 1.00], np.float32),
            pad_top[..., None],
        ) * (0.95 + 0.06 * SNOW_TEX)[..., None]

        d_sh = ((X - sx_c) / 0.142) ** 2 + ((Y - 0.312) / 0.028) ** 2
        sh_amt = np.exp(-d_sh * 1.9) * 0.85
        pad_col = pad_col * (1.0 - sh_amt)[..., None]

        g_col = mix(g_col, pad_col, m_pad[..., None])
        g_mask = np.maximum(g_mask, m_pad)

    BASE_BG = mix(bg, g_col, m_gnd[..., None]).astype(np.float32)
    BASE_SOLID = np.maximum(solid, g_mask * m_gnd).astype(np.float32)
    SM_RS, SM_CS = roi(0.0, 0.02, 0.38, 0.34)
    reset()


def reset():
    S["last"] = 0.0
    S["settled_sm"] = None
    S["settled_sm_mask"] = None


def render_aurora_sky(img, t):
    cy1 = -0.330 + 0.085 * np.sin(XS_SKY * 4.2 - t * 0.58) + 0.028 * np.cos(XS_SKY * 7.8 + t * 0.34)
    w1 = 0.038 + 0.014 * np.sin(XS_SKY * 4.8 + t * 0.68)
    z1 = (YS_SKY - cy1) / w1
    b1 = 1.0 / (1.0 + z1 * z1)
    sh1 = 0.50 + 0.50 * np.sin(XS_SKY * 13.0 - t * 1.35 + YS_SKY * 7.0)
    g1 = b1 * b1 * sh1 * (0.72 + 0.28 * np.cos(XS_SKY * 2.4 - t * 0.38))

    cy2 = -0.215 + 0.068 * np.cos(XS_SKY * 3.7 + t * 0.46) - 0.024 * np.sin(XS_SKY * 6.9 - t * 0.48)
    w2 = 0.044 + 0.013 * np.cos(XS_SKY * 4.1 - t * 0.55)
    z2 = (YS_SKY - cy2) / w2
    b2 = 1.0 / (1.0 + z2 * z2)
    sh2 = 0.48 + 0.52 * np.sin(XS_SKY * 10.5 + t * 1.05 - YS_SKY * 6.0)
    g2 = b2 * b2 * sh2 * (0.70 + 0.30 * np.sin(XS_SKY * 2.2 + t * 0.30))

    img[:GND_ROW0, :] += (g1[..., None] * AUR_C1 + g2[..., None] * AUR_C2) * SKY_FADE[..., None]
    img[MOON_RS, MOON_CS] = img[MOON_RS, MOON_CS] * (1.0 - MOON_ALPHA) + MOON_COL
    return img


def render_pine(img, solid_mask, px, t, tx, base_y, height, width, n_tiers=5, phase=0.0, depth_dim=1.0):
    c0 = max(0, int((tx - width * 1.18 - X0_MIN) / DX_PIX))
    c1 = min(GW, int((tx + width * 1.18 - X0_MIN) / DX_PIX) + 1)
    if c1 <= c0:
        return img, solid_mask

    r_tr0 = max(0, int((base_y - height * 0.05 - Y0_MIN) / DY_PIX))
    r_tr1 = min(GH, int((base_y + height * 0.02 - Y0_MIN) / DY_PIX) + 1)
    if r_tr1 > r_tr0:
        Xs = X[r_tr0:r_tr1, c0:c1]
        Ys = Y[r_tr0:r_tr1, c0:c1]
        trunk_d = sd_box(Xs - tx, Ys - (base_y - height * 0.015), width * 0.09, height * 0.025, 0.002)
        m_tr = np.clip(-trunk_d / (1.5 * px) + 0.5, 0.0, 1.0)
        img[r_tr0:r_tr1, c0:c1] = mix(img[r_tr0:r_tr1, c0:c1], TRUNK_COL * depth_dim, m_tr[..., None])
        solid_mask[r_tr0:r_tr1, c0:c1] = np.maximum(solid_mask[r_tr0:r_tr1, c0:c1], m_tr)

    for k in range(n_tiers):
        fk = k / max(1, n_tiers - 1)
        tier_bot = base_y - height * (0.02 + 0.72 * fk)
        tier_top = tier_bot - height * (0.31 - 0.04 * fk)
        tier_h = tier_bot - tier_top
        tier_w = width * (1.0 - 0.54 * fk)

        r0 = max(0, int((tier_top - 0.01 - Y0_MIN) / DY_PIX))
        r1 = min(GH, int((tier_bot + 0.04 * height - Y0_MIN) / DY_PIX) + 1)
        if r1 <= r0:
            continue

        Xs = X[r0:r1, c0:c1]
        Ys = Y[r0:r1, c0:c1]
        nt = NEEDLE_TEX[r0:r1, c0:c1]

        sway = 0.0050 * m.sin(t * 1.35 + phase + k * 0.35) * ((0.20 + 0.80 * fk) ** 1.3)
        dx = Xs - (tx + sway)
        rel_y = (Ys - tier_top) / max(tier_h, 1e-4)
        half_w = tier_w * (np.clip(rel_y, 0.0, 1.22) ** 1.05)

        scallop = 0.032 * (np.cos(dx * 58.0 + k * 2.2) * 0.5 + 0.5)
        bot_lim = tier_bot + scallop * height
        in_y = (Ys >= tier_top) & (Ys <= bot_lim)
        edge_d = half_w - np.abs(dx)
        vert_d = np.minimum(Ys - tier_top, bot_lim - Ys)
        d_tier = np.where(in_y, np.minimum(edge_d, vert_d), -0.05)
        m_tier = np.clip(d_tier / (1.5 * px) + 0.5, 0.0, 1.0)

        under_shadow = ss(0.0, 0.28, rel_y) if k < n_tiers - 1 else 1.0
        needle_val = (0.48 + 0.52 * nt) * under_shadow
        lit_side = ss(-tier_w * 0.4, tier_w * 0.6, dx)
        col_t = mix(PINE_DEEP, PINE_MID, np.clip(needle_val, 0.0, 1.0)[..., None])
        col_t = mix(col_t, PINE_LIT, np.clip(needle_val * lit_side, 0.0, 1.0)[..., None])

        norm_x = dx / np.maximum(half_w, 0.015)
        slope_wave = 0.14 * np.sin(rel_y * 7.5 + nt * 3.0)
        right_drape = ss(0.10 - 0.34 * (1.0 - rel_y) + slope_wave, 0.44 - 0.24 * (1.0 - rel_y) + slope_wave, norm_x)
        left_drape = ss(-0.56 - 0.14 * rel_y + slope_wave * 0.6, -0.82 - 0.05 * rel_y + slope_wave * 0.6, norm_x) * ss(0.20, 0.72, rel_y)
        apex_cap = ss(0.32, 0.04, rel_y) * (0.78 + 0.22 * nt)
        bough_tip_snow = ss(0.032 * (1.0 - 0.2 * fk), 0.0, np.maximum(edge_d, 0.0)) * ss(0.18, 0.58, rel_y)
        bottom_fringe = ss(0.74, 0.96, rel_y) * (0.58 + 0.42 * nt)
        snow_amt = np.clip(np.maximum.reduce([right_drape * 0.96, left_drape * 0.88, apex_cap, bough_tip_snow * 0.85, bottom_fringe]), 0.0, 1.0)
        snow_amt = snow_amt * (0.78 + 0.22 * nt)

        frost_col = mix(FROST_COOL, FROST_WHITE, np.clip(0.35 + 0.65 * lit_side, 0.0, 1.0)[..., None])
        col_t = mix(col_t, frost_col, snow_amt[..., None]) * depth_dim

        img[r0:r1, c0:c1] = mix(img[r0:r1, c0:c1], col_t, m_tier[..., None])
        solid_mask[r0:r1, c0:c1] = np.maximum(solid_mask[r0:r1, c0:c1], m_tier)

    return img, solid_mask


def shade_snowball(img, solid_mask, px, cx, cy, rx, ry, grow_k, ao_top_y=None):
    if grow_k <= 0.002:
        return img, solid_mask
    sx = rx * grow_k
    sy = ry * grow_k
    rs, cs = roi(cx, cy, sx + 0.02, sy + 0.02)
    Xs, Ys = X[rs, cs], Y[rs, cs]

    d = sd_ellipse(Xs, Ys, cx, cy, sx, sy)
    m_b = np.clip(-d / (1.5 * px) + 0.5, 0.0, 1.0)

    nx = np.clip((Xs - cx) / max(sx, 1e-4), -1.0, 1.0)
    ny = np.clip((Ys - cy) / max(sy, 1e-4), -1.0, 1.0)
    nz = np.sqrt(np.maximum(0.04, 1.0 - nx * nx - ny * ny))

    diff = np.clip(0.24 + nz * 0.66 + nx * 0.24 - ny * 0.14, 0.0, 1.0)

    shadow_blue = np.array([0.22, 0.58, 1.00], np.float32)
    ice_blue = np.array([0.66, 0.90, 1.00], np.float32)
    snow_white = np.array([1.00, 1.00, 1.00], np.float32)

    col = mix(shadow_blue, ice_blue, ss(0.16, 0.48, diff)[..., None])
    col = mix(col, snow_white, ss(0.46, 0.80, diff)[..., None])
    col = col * SNOW_TEX[rs, cs][..., None]

    if ao_top_y is not None:
        ao = 1.0 - 0.20 * np.exp(-((Ys - ao_top_y) / (sy * 0.30)) ** 2) * ss(sx * 0.88, 0.0, np.abs(Xs - cx))
        col = col * ao[..., None]

    img[rs, cs] = mix(img[rs, cs], col, m_b[..., None])
    solid_mask[rs, cs] = np.maximum(solid_mask[rs, cs], m_b)
    return img, solid_mask


def draw_twig_arms(img, solid_mask, body_mask, px, t, arm_k, hat_k, cx_b, cx_g):
    if arm_k <= 0.01:
        return img, solid_mask
    wave_l = 0.012 * m.sin(t * 2.1)
    wave_r = 0.012 * m.sin(t * 2.1 + 1.1)

    wood_dark = np.array([0.64, 0.34, 0.15], np.float32)
    wood_light = np.array([0.96, 0.62, 0.30], np.float32)

    segments = [
        (cx_b - 0.102, 0.025, cx_b - 0.215, -0.068 + wave_l, 0.013),
        (cx_b - 0.215, -0.068 + wave_l, cx_b - 0.250, -0.106 + wave_l, 0.0085),
        (cx_b - 0.215, -0.068 + wave_l, cx_b - 0.258, -0.052 + wave_l, 0.0085),
        (cx_g + 0.102, 0.025, cx_g + 0.215, -0.068 + wave_r, 0.013),
        (cx_g + 0.215, -0.068 + wave_r, cx_g + 0.250, -0.106 + wave_r, 0.0085),
        (cx_g + 0.215, -0.068 + wave_r, cx_g + 0.258, -0.052 + wave_r, 0.0085),
        (cx_b + 0.100, 0.025, 0.0, 0.072, 0.013),
        (cx_g - 0.100, 0.025, 0.0, 0.072, 0.013),
    ]

    rs, cs = roi(0.0, -0.01, 0.50, 0.15)
    Xs, Ys = X[rs, cs], Y[rs, cs]
    sub = img[rs, cs]
    sub_m = solid_mask[rs, cs]
    sub_b = body_mask[rs, cs]

    for (ax, ay, bx, by, rad) in segments:
        ex = ax + (bx - ax) * arm_k
        ey = ay + (by - ay) * arm_k
        d = sd_capsule(Xs, Ys, ax, ay, ex, ey, rad)
        m_a = np.clip(-d / (1.3 * px) + 0.5, 0.0, 1.0) * np.clip(1.0 - sub_b * 1.4, 0.0, 1.0)
        shade = ss(-rad, rad, Ys - (ay + ey) * 0.5)
        col = mix(wood_light, wood_dark, shade[..., None])
        sub = mix(sub, col, m_a[..., None])
        sub_m = np.maximum(sub_m, m_a)

    if hat_k > 0.02:
        for (mx_, my_) in ((cx_b - 0.222, -0.076 + wave_l), (cx_g + 0.222, -0.076 + wave_r)):
            dm = np.sqrt((Xs - mx_) ** 2 + (Ys - my_) ** 2) - 0.021 * hat_k
            mm = np.clip(-dm / (1.3 * px) + 0.5, 0.0, 1.0)
            mcol = np.array([1.00, 0.48, 0.76], np.float32) if mx_ > 0 else np.array([0.95, 0.24, 0.26], np.float32)
            sub = mix(sub, mcol, mm[..., None])
            sub_m = np.maximum(sub_m, mm)

    img[rs, cs] = sub
    solid_mask[rs, cs] = sub_m
    return img, solid_mask


def draw_snowman_face(img, px, cx, hy, nose_dir, face_k, happy_k):
    if face_k <= 0.01:
        return img
    rs, cs = roi(cx, hy, 0.12, 0.10)
    Xs, Ys = X[rs, cs], Y[rs, cs]
    sub = img[rs, cs]

    eye_dx = 0.028
    eye_y = hy - 0.015
    for sx in (-1.0, 1.0):
        ex = cx + sx * eye_dx + nose_dir * 0.006
        if happy_k > 0.35:
            arc_y = eye_y + 0.014 * (((Xs - ex) / 0.016) ** 2 - 0.4)
            d_arc = np.maximum(np.abs(Ys - arc_y) - 0.0054, np.abs(Xs - ex) - 0.016)
            m_eye = np.clip(-d_arc / (1.2 * px) + 0.5, 0.0, 1.0) * face_k
        else:
            d_eye = sd_ellipse(Xs, Ys, ex, eye_y, 0.0155, 0.0145)
            m_eye = np.clip(-d_eye / (1.2 * px) + 0.5, 0.0, 1.0) * face_k
        sub = mix(sub, np.array([0.0, 0.0, 0.0], np.float32), m_eye[..., None])

    for sx in (-1.0, 1.0):
        ckx = cx + sx * 0.043 + nose_dir * 0.004
        cky = hy + 0.010
        d_ck = ((Xs - ckx) / 0.019) ** 2 + ((Ys - cky) / 0.011) ** 2
        m_ck = np.exp(-d_ck * 1.45) * 0.85 * face_k
        sub = mix(sub, np.array([1.00, 0.44, 0.62], np.float32), m_ck[..., None])

    for i in range(-2, 3):
        ang = i * 0.36
        mx_ = cx + nose_dir * 0.006 + m.sin(ang) * 0.030
        my_ = hy + 0.024 + m.cos(ang) * 0.007
        d_m = np.sqrt((Xs - mx_) ** 2 + (Ys - my_) ** 2) - 0.0060
        m_m = np.clip(-d_m / (1.2 * px) + 0.5, 0.0, 1.0) * face_k
        sub = mix(sub, np.array([0.0, 0.0, 0.0], np.float32), m_m[..., None])

    nx0 = cx + nose_dir * 0.006
    ny0 = hy + 0.004
    nx1 = nx0 + nose_dir * 0.054 * face_k
    ny1 = ny0 + 0.004
    pax, pay = Xs - nx0, Ys - ny0
    bax, bay = nx1 - nx0, ny1 - ny0
    h = np.clip((pax * bax + pay * bay) / (bax * bax + bay * bay + 1e-9), 0.0, 1.0)
    rad = 0.0125 * (1.0 - 0.78 * h)
    d_nose = np.sqrt((pax - bax * h) ** 2 + (pay - bay * h) ** 2) - rad
    m_nose = np.clip(-d_nose / (1.2 * px) + 0.5, 0.0, 1.0) * face_k
    nose_col = mix(
        np.array([1.00, 0.46, 0.05], np.float32),
        np.array([1.00, 0.78, 0.25], np.float32),
        ss(-0.005, 0.005, ny0 - Ys)[..., None],
    )
    sub = mix(sub, nose_col, m_nose[..., None])
    img[rs, cs] = sub
    return img


def draw_buttons(img, px, cx, ty, btn_k):
    if btn_k <= 0.01:
        return img
    rs, cs = roi(cx, ty + 0.01, 0.04, 0.09)
    Xs, Ys = X[rs, cs], Y[rs, cs]
    sub = img[rs, cs]
    for idx_b, by_off in enumerate((-0.028, 0.008, 0.044)):
        bk = ease((btn_k - idx_b * 0.18) / 0.55)
        if bk <= 0.01:
            continue
        bx = cx + (0.005 if cx < 0 else -0.005)
        by = ty + by_off
        d_b = np.sqrt((Xs - bx) ** 2 + (Ys - by) ** 2) - 0.0175 * bk
        m_b = np.clip(-d_b / (1.2 * px) + 0.5, 0.0, 1.0)
        sub = mix(sub, np.array([0.0, 0.0, 0.0], np.float32), m_b[..., None])
    img[rs, cs] = sub
    return img


def draw_scarves(img, solid_mask, px, t, scarf_k, cx_b, cx_g):
    if scarf_k <= 0.01:
        return img, solid_mask
    flut = 0.005 * m.sin(t * 2.8)
    neck_y = -0.032

    rs, cs = roi(cx_b, neck_y + 0.05, 0.12, 0.14)
    Xs, Ys = X[rs, cs], Y[rs, cs]
    d_wrap_b = sd_capsule(Xs, Ys, cx_b - 0.078, neck_y - 0.004, cx_b + 0.078, neck_y - 0.004, 0.021 * scarf_k)
    d_tail_b = sd_capsule(Xs, Ys, cx_b + 0.038, neck_y + 0.004, cx_b + 0.044 + flut, neck_y + 0.125 * scarf_k, 0.019 * scarf_k)
    d_b = np.minimum(d_wrap_b, d_tail_b)
    m_b = np.clip(-d_b / (1.4 * px) + 0.5, 0.0, 1.0)
    stripe = np.sin((Xs - cx_b) * 68.0 + (Ys - neck_y) * 65.0) > 0.0
    col_b = np.where(
        stripe[..., None],
        np.array([1.00, 0.14, 0.16], np.float32),
        np.array([0.12, 0.90, 0.38], np.float32),
    ) * (0.82 + 0.18 * ss(-0.07, 0.07, Xs - cx_b))[..., None]
    img[rs, cs] = mix(img[rs, cs], col_b, m_b[..., None])
    solid_mask[rs, cs] = np.maximum(solid_mask[rs, cs], m_b)

    rs, cs = roi(cx_g, neck_y + 0.05, 0.12, 0.14)
    Xs, Ys = X[rs, cs], Y[rs, cs]
    d_wrap_g = sd_capsule(Xs, Ys, cx_g - 0.078, neck_y - 0.004, cx_g + 0.078, neck_y - 0.004, 0.021 * scarf_k)
    d_tail_g = sd_capsule(Xs, Ys, cx_g - 0.032, neck_y + 0.004, cx_g - 0.038 - flut, neck_y + 0.118 * scarf_k, 0.018 * scarf_k)
    d_g = np.minimum(d_wrap_g, d_tail_g)
    m_g = np.clip(-d_g / (1.4 * px) + 0.5, 0.0, 1.0)
    stripe_g = np.sin((Xs - cx_g) * 65.0 + (Ys - neck_y) * 65.0) > 0.30
    col_g = np.where(
        stripe_g[..., None],
        np.array([1.00, 0.86, 0.96], np.float32),
        np.array([1.00, 0.44, 0.74], np.float32),
    ) * (0.82 + 0.18 * ss(-0.07, 0.07, Xs - cx_g))[..., None]
    img[rs, cs] = mix(img[rs, cs], col_g, m_g[..., None])
    solid_mask[rs, cs] = np.maximum(solid_mask[rs, cs], m_g)

    return img, solid_mask


def draw_hats(img, solid_mask, px, hat_k, cx_b, cx_g, hy):
    if hat_k <= 0.01:
        return img, solid_mask
    drop = (1.0 - hat_k) * 0.13

    brim_y = hy - 0.062 - drop
    rs, cs = roi(cx_b, brim_y - 0.035, 0.12, 0.10)
    Xs, Ys = X[rs, cs], Y[rs, cs]
    d_brim = sd_box(Xs - cx_b, Ys - brim_y, 0.098, 0.012, 0.005)
    d_crown = sd_box(Xs - cx_b, Ys - (brim_y - 0.050), 0.060, 0.048, 0.005)
    d_hat = np.minimum(d_brim, d_crown)
    m_hat = np.clip(-d_hat / (1.4 * px) + 0.5, 0.0, 1.0) * hat_k

    hi_side = ss(-0.01, 0.06, Xs - cx_b) * 0.36
    edge_outline = ss(0.018, 0.0, np.abs(d_hat)) * 0.32
    hat_col = np.array([0.38, 0.44, 0.56], np.float32) + (hi_side + edge_outline)[..., None] * np.array([0.52, 0.60, 0.74], np.float32)
    band_m = (np.abs(Ys - (brim_y - 0.019)) < 0.010) & (np.abs(Xs - cx_b) < 0.062)
    hat_col = np.where(band_m[..., None], np.array([1.00, 0.20, 0.24], np.float32), hat_col)
    img[rs, cs] = mix(img[rs, cs], hat_col, m_hat[..., None])
    solid_mask[rs, cs] = np.maximum(solid_mask[rs, cs], m_hat)

    hb_y = hy - 0.056 - drop
    rs, cs = roi(cx_g, hb_y - 0.01, 0.13, 0.09)
    Xs, Ys = X[rs, cs], Y[rs, cs]
    sub = img[rs, cs]
    sub_m = solid_mask[rs, cs]

    arc_y = hb_y + 0.018 + 0.018 * (((Xs - cx_g) / 0.065) ** 2 - 0.5)
    d_band = np.maximum(np.abs(Ys - arc_y) - 0.013, np.abs(Xs - cx_g) - 0.062)
    m_band = np.clip(-d_band / (1.4 * px) + 0.5, 0.0, 1.0) * hat_k
    sub = mix(sub, np.array([1.00, 0.48, 0.76], np.float32), m_band[..., None])
    sub_m = np.maximum(sub_m, m_band)

    for sx in (-1.0, 1.0):
        px_c = cx_g + sx * 0.058
        py_c = hb_y - 0.008
        pr = 0.042
        d_pom = np.sqrt((Xs - px_c) ** 2 + (Ys - py_c) ** 2) - pr
        m_pom = np.clip(-d_pom / (1.4 * px) + 0.5, 0.0, 1.0) * hat_k
        nx = np.clip((Xs - px_c) / pr, -1.0, 1.0)
        ny = np.clip((Ys - py_c) / pr, -1.0, 1.0)
        nz = np.sqrt(np.maximum(0.04, 1.0 - nx * nx - ny * ny))
        diff = np.clip(0.32 + nz * 0.55 + nx * 0.20 - ny * 0.20, 0.0, 1.0)
        pom_col = mix(np.array([0.88, 0.36, 0.64], np.float32), np.array([1.00, 0.86, 0.95], np.float32), diff[..., None])
        sub = mix(sub, pom_col, m_pom[..., None])
        sub_m = np.maximum(sub_m, m_pom)

    img[rs, cs] = sub
    solid_mask[rs, cs] = sub_m
    return img, solid_mask


def draw_floating_heart(img, solid_mask, px, t, heart_k):
    if heart_k <= 0.005:
        return img, solid_mask
    pulse = 1.0 + 0.08 * m.sin(t * 4.2)
    bob = 0.005 * m.sin(t * 2.4)
    sz = 0.056 * heart_k * pulse
    cy = -0.022 + bob
    rs, cs = roi(0.0, cy, sz * 2.4, sz * 2.4)
    Xs, Ys = X[rs, cs], Y[rs, cs]

    hx = Xs / max(sz, 1e-4)
    hy = -(Ys - cy) / max(sz, 1e-4) + 0.14
    f = (hx * hx + hy * hy - 1.0) ** 3 - hx * hx * (hy ** 3)
    q = 0.5 * np.cbrt(f) * sz
    m_h = np.clip(-q / (1.6 * px) + 0.5, 0.0, 1.0)
    halo = np.exp(-np.clip(q / max(sz, 1e-4), 0.0, 6.0) * 2.6) * 0.52 * heart_k

    heart_col = mix(
        np.array([1.00, 0.32, 0.66], np.float32),
        np.array([1.00, 0.82, 0.94], np.float32),
        np.clip(-q / max(sz * 0.45, 1e-4), 0.0, 1.0)[..., None],
    )
    sub = img[rs, cs] + halo[..., None] * np.array([1.00, 0.34, 0.68], np.float32)
    img[rs, cs] = mix(sub, heart_col, m_h[..., None])
    solid_mask[rs, cs] = np.maximum(solid_mask[rs, cs], m_h)
    return img, solid_mask


def stamp_stars(idx, fg, sky_mask, t, fade_k):
    twk = 0.45 + 0.55 * (np.sin(t * 2.5 + STP * 12.0) ** 2)
    c = ((W * 0.5 + STX * H) / CW).astype(np.int32)
    r = ((H * 0.5 + STY * H) / CH).astype(np.int32)
    ok = (c >= 0) & (c < COLS) & (r >= 0) & (r < ROWS)
    if not ok.any():
        return
    ii = np.flatnonzero(ok)
    c, r = c[ii], r[ii]
    mask = (sky_mask[r, c] > 0.7) & (~MOON_MASK[r, c]) & (YC[r, c] < ground_height(XC[r, c]) - 0.004)
    ii, c, r = ii[mask], c[mask], r[mask]
    if len(ii) == 0:
        return
    ch = np.where(STT[ii] < 0.6, CIDX["."], np.where(STT[ii] < 0.9, CIDX["+"], CIDX["*"]))
    idx[r, c] = ch
    fg[r, c] = np.array([0.88, 0.95, 1.0], np.float32)[None, :] * (twk[ii] * fade_k)[:, None]


def stamp_ground_dots(idx, fg, solid_c, t, fade_k):
    ok = (solid_c[GR_IDX, GC_IDX] < 0.25) & (idx[GR_IDX, GC_IDX] == 0)
    if not ok.any():
        return
    r = GR_IDX[ok]
    c = GC_IDX[ok]
    wave = 0.5 + 0.5 * np.sin(GND_WAVE_POS[ok] - t * 0.95)
    glit_v = np.maximum(0.0, np.sin(t * 1.6 + GND_GLIT_SEED[ok] * 25.0) - 0.78) * 4.5
    glit = (glit_v * glit_v) * GND_HIGH[ok]
    br = np.clip((0.26 + 0.18 * wave + 0.58 * glit) * fade_k, 0.0, 0.95)
    idx[r, c] = np.where(glit > 0.30, CIDX[":"], CIDX["."])
    fg[r, c] = GND_DOT_COL[None, :] * br[:, None]


def stamp_snowflakes(idx, fg, idx_base, t, fade_k):
    sx = ((SN_X0 + SN_SWAY * np.sin(t * SN_FRQ + SN_PH) + 0.73) % 1.46) - 0.73
    sy = ((SN_Y0 + t * SN_SPD + 0.50) % 1.00) - 0.50
    c = ((W * 0.5 + sx * H) / CW).astype(np.int32)
    r = ((H * 0.5 + sy * H) / CH).astype(np.int32)
    ok = (c >= 0) & (c < COLS) & (r >= 0) & (r < ROWS)
    if not ok.any():
        return
    ii = np.flatnonzero(ok)
    c, r = c[ii], r[ii]
    keep = (idx_base[r, c] < 6) | (SN_KIND[ii] >= 0.78)
    ii, c, r = ii[keep], c[keep], r[keep]
    ch = np.where(SN_KIND[ii] < 0.45, CIDX["."], np.where(SN_KIND[ii] < 0.78, CIDX["+"], CIDX["*"]))
    idx[r, c] = ch
    fg[r, c] = np.array([0.92, 0.97, 1.0], np.float32) * fade_k


def bloom_cells(cells):
    b = np.clip(cells - 0.78, 0.0, None)
    s_ = (b * 2.0 + np.roll(b, 1, 0) + np.roll(b, -1, 0) + np.roll(b, 1, 1) + np.roll(b, -1, 1)) / 6.0
    s_ = (s_ * 2.0 + np.roll(s_, 1, 0) + np.roll(s_, -1, 0) + np.roll(s_, 1, 1) + np.roll(s_, -1, 1)) / 6.0
    return cells + 0.28 * s_


def render_cells(t, dt):
    S["last"] = t
    px = 1.0 / GH
    fade_in = 0.15 + 0.85 * ease(t / 0.70)

    cx_b, cx_g = -0.210, 0.210
    k_base = ease_back((t - T_BODY0) / 0.55)
    k_mid = ease_back((t - (T_BODY0 + 0.40)) / 0.55)
    k_head = ease_back((t - (T_BODY0 + 0.80)) / 0.55)
    k_face = ease((t - T_FACE0) / 0.65)
    k_btn = ease((t - (T_FACE0 + 0.85)) / 0.80)
    k_arms = ease((t - T_ARMS0) / 1.15)
    k_scarf = ease((t - T_SCARF0) / 1.25)
    k_hat = ease_back((t - T_HAT0) / 1.00)
    k_heart = ease_back((t - T_HEART0) / 1.00)

    bt = (t - 7.8) % 4.5
    happy_k = 1.0 if (t > 9.0 and bt < 1.2) else 0.0

    img = BASE_BG.copy()
    solid_mask = BASE_SOLID.copy()
    img = render_aurora_sky(img, t)

    img, solid_mask = render_pine(img, solid_mask, px, t, tx=-0.695, base_y=0.235, height=0.35, width=0.125, n_tiers=5, phase=0.3, depth_dim=0.95)
    img, solid_mask = render_pine(img, solid_mask, px, t, tx=-0.468, base_y=0.195, height=0.30, width=0.115, n_tiers=5, phase=1.6, depth_dim=0.88)
    img, solid_mask = render_pine(img, solid_mask, px, t, tx=0.525, base_y=0.210, height=0.34, width=0.138, n_tiers=5, phase=0.8, depth_dim=0.98)
    img, solid_mask = render_pine(img, solid_mask, px, t, tx=0.735, base_y=0.230, height=0.28, width=0.105, n_tiers=4, phase=2.2, depth_dim=0.88)

    settled = (k_base >= 1.0 and k_mid >= 1.0 and k_head >= 1.0 and k_face >= 1.0 and k_btn >= 1.0 and k_hat >= 1.0 and happy_k == 0.0)
    if settled and S["settled_sm"] is not None:
        sm_sub = S["settled_sm"]
        sm_msub = S["settled_sm_mask"]
        sm_m = np.zeros_like(solid_mask)
        sm_m[SM_RS, SM_CS] = sm_msub
        img[SM_RS, SM_CS] = mix(img[SM_RS, SM_CS], sm_sub, sm_msub[..., None])
        solid_mask[SM_RS, SM_CS] = np.maximum(solid_mask[SM_RS, SM_CS], sm_msub)
    else:
        sm_img = np.zeros_like(img)
        sm_m = np.zeros_like(solid_mask)
        hy = -0.095
        sm_img, sm_m = shade_snowball(sm_img, sm_m, px, cx_b, 0.210, 0.140, 0.114, k_base)
        sm_img, sm_m = shade_snowball(sm_img, sm_m, px, cx_g, 0.210, 0.140, 0.114, k_base)
        sm_img, sm_m = shade_snowball(sm_img, sm_m, px, cx_b, 0.042, 0.106, 0.088, k_mid, ao_top_y=0.108)
        sm_img, sm_m = shade_snowball(sm_img, sm_m, px, cx_g, 0.042, 0.106, 0.088, k_mid, ao_top_y=0.108)
        sm_img, sm_m = shade_snowball(sm_img, sm_m, px, cx_b, hy, 0.080, 0.070, k_head, ao_top_y=-0.036)
        sm_img, sm_m = shade_snowball(sm_img, sm_m, px, cx_g, hy, 0.080, 0.070, k_head, ao_top_y=-0.036)

        sm_img = draw_buttons(sm_img, px, cx_b, 0.044, k_btn)
        sm_img = draw_buttons(sm_img, px, cx_g, 0.044, k_btn)
        sm_img = draw_snowman_face(sm_img, px, cx_b, hy, nose_dir=+1.0, face_k=k_face, happy_k=happy_k)
        sm_img = draw_snowman_face(sm_img, px, cx_g, hy, nose_dir=-1.0, face_k=k_face, happy_k=happy_k)
        sm_img, sm_m = draw_hats(sm_img, sm_m, px, k_hat, cx_b, cx_g, hy)

        if settled:
            S["settled_sm"] = sm_img[SM_RS, SM_CS].copy()
            S["settled_sm_mask"] = sm_m[SM_RS, SM_CS].copy()
        img[SM_RS, SM_CS] = mix(img[SM_RS, SM_CS], sm_img[SM_RS, SM_CS], sm_m[SM_RS, SM_CS][..., None])
        solid_mask[SM_RS, SM_CS] = np.maximum(solid_mask[SM_RS, SM_CS], sm_m[SM_RS, SM_CS])

    img, solid_mask = draw_scarves(img, solid_mask, px, t, k_scarf, cx_b, cx_g)
    img, solid_mask = draw_twig_arms(img, solid_mask, sm_m, px, t, k_arms, k_hat, cx_b, cx_g)
    img, solid_mask = draw_floating_heart(img, solid_mask, px, t, k_heart)

    if SS > 1:
        cells = img.reshape(ROWS, SS, COLS, SS, 3).mean((1, 3))
        solid_c = solid_mask.reshape(ROWS, SS, COLS, SS).mean((1, 3))
    else:
        cells = img
        solid_c = solid_mask
    sky_mask = 1.0 - np.clip(solid_c * 2.0, 0.0, 1.0)

    cells = bloom_cells(cells)
    peak = np.maximum(cells, 0.0)
    lum_c = 0.30 * peak[..., 0] + 0.60 * peak[..., 1] + 0.10 * peak[..., 2]
    cells = np.clip(peak / (1.0 + 0.12 * lum_c)[..., None], 0.0, 1.0) * fade_in

    lum = 0.30 * cells[..., 0] + 0.60 * cells[..., 1] + 0.10 * cells[..., 2]
    cmax = cells.max(axis=-1)
    li = np.clip(0.45 * lum + 0.55 * cmax, 0.0, 1.0) ** 0.68

    ri = np.clip((li + (HSC - 0.5) * 0.08) * NR, 0.0, NR).astype(np.int32)
    solid_bright = (solid_c > 0.45) & (cmax > 0.08)
    ri = np.where(solid_bright, np.maximum(ri, SOLID_FLOOR), ri)
    ri = np.where(cmax < 0.04, 0, ri)

    idx = ri.copy()
    fg = np.clip(cells * 1.22, 0.0, 1.0).astype(np.float32)

    stamp_ground_dots(idx, fg, solid_c, t, fade_in)
    stamp_stars(idx, fg, sky_mask, t, fade_in)
    stamp_snowflakes(idx, fg, ri, t, fade_in)

    q = np.clip((np.clip(fg, 0.0, 1.0) * (QL - 0.01)).astype(np.int16), 0, QL - 1)
    return idx, q


def render_ppm_bytes(idx, q):
    fg_u8 = Q_TO_U8[q]
    rgb = (GLYPH_MASK[idx] * fg_u8[:, :, None, None, :] + GLYPH_BG[idx]).transpose(0, 2, 1, 3, 4).reshape(H, W, 3)
    return PPM_HDR + rgb.tobytes()


def main():
    import tkinter as tk

    root = tk.Tk()
    root.title("Snowman & Snowgirl — Mahamad (aGsx)")
    root.resizable(False, False)
    root.bind("<Escape>", lambda e: root.destroy())
    init(CW_GLYPH, CH_GLYPH, MAXROWS)
    scr_w, scr_h = root.winfo_screenwidth(), root.winfo_screenheight()
    root.geometry(f"{W}x{H}+{(scr_w - W) // 2}+{max(0, (scr_h - H) // 2 - 20)}")
    cv = tk.Canvas(root, width=W, height=H, bg="#010203", highlightthickness=0)
    cv.pack()

    photo = tk.PhotoImage(width=W, height=H)
    cv.create_image(0, 0, anchor="nw", image=photo)
    st = {"t0": time.perf_counter(), "paused": False, "pause_t": 0.0}

    def draw_at(t_val, dt_val=0.0):
        idx, q = render_cells(t_val, dt_val)
        photo.configure(data=render_ppm_bytes(idx, q))

    def restart(_=None):
        st["t0"] = time.perf_counter()
        st["pause_t"] = 0.0
        st["paused"] = False
        reset()
        draw_at(0.0, 0.0)

    def toggle_pause(_=None):
        now = time.perf_counter()
        if st["paused"]:
            st["t0"] = now - st["pause_t"]
            S["last"] = st["pause_t"]
            st["paused"] = False
        else:
            st["pause_t"] = now - st["t0"]
            st["paused"] = True

    def go_final(_=None):
        now = time.perf_counter()
        st["t0"] = now - T_FINAL
        st["pause_t"] = T_FINAL
        S["last"] = T_FINAL
        draw_at(T_FINAL, 0.0)

    root.bind("r", restart)
    root.bind("R", restart)
    root.bind("<space>", toggle_pause)
    root.bind("a", go_final)
    root.bind("A", go_final)

    def tick():
        if st["paused"]:
            root.after(40, tick)
            return
        t0 = time.perf_counter()
        t = t0 - st["t0"]
        dt = min(max(t - S["last"], 0.0), 0.1)
        draw_at(t, dt)
        el = (time.perf_counter() - t0) * 1000
        root.after(max(1, int(1000 / FPS - el)), tick)

    root.after(50, tick)
    root.mainloop()


if __name__ == "__main__":
    main()
