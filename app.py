# -*- coding: utf-8 -*-
"""Block Blast Solver API — bản sửa lỗi nhận diện hình học.

Điểm sửa chính so với bản cũ:
  1. locate_board(): quét "comb 9 vạch" trên TOÀN ảnh (vector hoá bằng numpy),
     bỏ hard-limit by <= 0.36h  ->  hết bị lệch lên 2 ô.
  2. detect_pieces(): tự tìm 3 khối chờ bằng năng lượng màu ở nửa dưới ảnh,
     KHÔNG còn suy ra từ by + bw  ->  nhận được khối xếp chờ.
  3. read_piece_grid(): suy kích thước khối bằng "rãnh tối" giữa các ô
     (seam valley) thay vì magic number.
  4. render: nới canvas ra 4 phía, nhãn Hàng/Cột dùng anchor nên không lệch.
"""

import io
import os
import json
import time
import base64
import hashlib
import itertools

import numpy as np
from PIL import Image, ImageDraw, ImageFont
from flask import Flask, request, send_file, jsonify, Response

# ==============================================================================
# CẤU HÌNH
# ==============================================================================
app = Flask(__name__)
app.config["MAX_CONTENT_LENGTH"] = 16 * 1024 * 1024

DATA_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "solver_data")
os.makedirs(DATA_DIR, exist_ok=True)
HISTORY_LOG_PATH = os.path.join(DATA_DIR, "solver_history.jsonl")

TARGET_W = 720
SLOT_COLORS = [(255, 82, 82), (76, 217, 100), (64, 156, 255)]
SLOT_NAMES = ["TRÁI", "GIỮA", "PHẢI"]

FULL_MASK = 0xFFFFFFFFFFFFFFFF
NOT_RIGHT = 0x7F7F7F7F7F7F7F7F
ROW_MASKS = [0xFF << (r * 8) for r in range(8)]
COL_MASKS = [0x0101010101010101 << c for c in range(8)]

_FONT_CACHE = {}


def get_font(size):
    if size in _FONT_CACHE:
        return _FONT_CACHE[size]
    font = None
    for path in (
        "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf",
        "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
        "/System/Library/Fonts/Supplemental/Arial Bold.ttf",
    ):
        try:
            font = ImageFont.truetype(path, size)
            break
        except Exception:
            font = None
    if font is None:
        font = ImageFont.load_default()
    _FONT_CACHE[size] = font
    return font


# ==============================================================================
# 0. TIỆN ÍCH ẢNH
# ==============================================================================
def compute_image_sha256(pil_img):
    return hashlib.sha256(pil_img.convert("RGB").tobytes()).hexdigest()


def normalize_image(pil_img, target_w=TARGET_W):
    rgb = pil_img.convert("RGB")
    if rgb.width != target_w:
        scale = target_w / float(rgb.width)
        rgb = rgb.resize((target_w, max(1, int(round(rgb.height * scale)))),
                         Image.Resampling.LANCZOS)
    return rgb, np.asarray(rgb, dtype=np.float32)


def _luma(arr):
    return 0.299 * arr[:, :, 0] + 0.587 * arr[:, :, 1] + 0.114 * arr[:, :, 2]


def _chroma(arr):
    return arr.max(axis=2) - arr.min(axis=2)


def _otsu(values):
    v = np.asarray(values, dtype=np.float64).ravel()
    if v.size == 0:
        return 0.0
    lo, hi = float(v.min()), float(v.max())
    if hi - lo < 1e-6:
        return hi
    hist, edges = np.histogram(v, bins=64, range=(lo, hi))
    p = hist.astype(np.float64)
    p /= p.sum()
    centres = (edges[:-1] + edges[1:]) * 0.5
    omega = np.cumsum(p)
    mu = np.cumsum(p * centres)
    mu_t = mu[-1]
    denom = omega * (1.0 - omega)
    denom[denom <= 1e-12] = 1e-12
    sigma_b = (mu_t * omega - mu) ** 2 / denom
    return float(centres[int(np.argmax(sigma_b))])


def _group_runs(flags, min_gap=3):
    flags = np.asarray(flags, dtype=bool)
    runs, start = [], None
    for i in range(flags.size):
        if flags[i] and start is None:
            start = i
        elif not flags[i] and start is not None:
            runs.append((start, i - 1))
            start = None
    if start is not None:
        runs.append((start, flags.size - 1))
    merged = []
    for a, b in runs:
        if merged and a - merged[-1][1] <= min_gap:
            merged[-1] = (merged[-1][0], b)
        else:
            merged.append((a, b))
    return merged


def _dilate(prof, rad=2):
    out = prof.astype(np.float64).copy()
    for k in range(1, rad + 1):
        if k >= out.size:
            break
        out[k:] = np.maximum(out[k:], prof[:-k])
        out[:-k] = np.maximum(out[:-k], prof[k:])
    return out


# ==============================================================================
# 1. ĐỊNH VỊ BÀN CỜ — quét comb 9 vạch, vector hoá, KHÔNG hard-limit
# ==============================================================================
def _best_comb(prof, length, n=9):
    """Tìm offset (0..length) có tổng n vạch cách đều lớn nhất."""
    lim = prof.shape[0] - length
    if lim <= 1:
        return 0, -1e18
    starts = np.arange(0, lim)
    step = length / float(n - 1)
    offs = np.round(np.arange(n) * step).astype(np.int64)
    idx = np.clip(starts[:, None] + offs[None, :], 0, prof.shape[0] - 1)
    vals = prof[idx].sum(axis=1) / float(n)
    i = int(np.argmax(vals))
    return int(starts[i]), float(vals[i])


def locate_board(arr):
    """Trả về (bx, by, bw) — bàn cờ vuông cạnh bw."""
    h, w, _ = arr.shape
    lum = _luma(arr)

    gx = np.zeros_like(lum)
    gx[:, 1:] = np.abs(np.diff(lum, axis=1))
    gy = np.zeros_like(lum)
    gy[1:, :] = np.abs(np.diff(lum, axis=0))

    # --- Bước 1: bề rộng + mép trái (chỉ dùng gradient cột) ---
    y0, y1 = int(h * 0.05), int(h * 0.92)
    col_raw = gx[y0:y1].mean(axis=0)
    m = col_raw.mean()
    if m > 1e-6:
        col_raw = col_raw / m
    col_prof = _dilate(col_raw, 2)

    best = (-1e18, int(round(w * 0.04)), int(round(w * 0.92)))
    for bw_f in np.arange(0.60, 1.0001, 0.01):
        bw_c = int(round(bw_f * w))
        if bw_c < w * 0.40 or bw_c >= w:
            continue
        step = bw_c / 8.0
        starts = np.arange(0, w - bw_c, 2)
        if starts.size == 0:
            continue
        offs = np.round(np.arange(9) * step).astype(np.int64)
        idx = np.clip(starts[:, None] + offs[None, :], 0, w - 1)
        vals = col_prof[idx].sum(axis=1) / 9.0
        vals = vals - np.abs((starts + bw_c * 0.5) - w * 0.5) / w * 0.20
        i = int(np.argmax(vals))
        if vals[i] > best[0]:
            best = (float(vals[i]), int(starts[i]), bw_c)

    _, bx0, bw0 = best

    # --- Bước 1b: tinh chỉnh cục bộ ---
    best = (-1e18, bx0, bw0)
    for bw_c in range(max(int(w * 0.40), bw0 - 12), min(w - 1, bw0 + 13)):
        step = bw_c / 8.0
        for bx_c in range(max(0, bx0 - 8), min(w - bw_c, bx0 + 9)):
            offs = np.round(np.arange(9) * step).astype(np.int64)
            ii = np.clip(bx_c + offs, 0, w - 1)
            s = float(col_prof[ii].mean())
            s -= abs((bx_c + bw_c * 0.5) - w * 0.5) / w * 0.20
            if s > best[0]:
                best = (s, bx_c, bw_c)
    _, bx, bw = best
    bx = int(max(0, min(bx, w - bw)))

    # --- Bước 2: mép trên, quét TOÀN chiều cao ảnh ---
    row_raw = gy[:, bx:bx + bw].mean(axis=1)
    m = row_raw.mean()
    if m > 1e-6:
        row_raw = row_raw / m
    row_prof = _dilate(row_raw, 2)

    max_by = max(1, h - bw - 1)
    step = bw / 8.0
    starts = np.arange(0, max_by)
    offs = np.round(np.arange(9) * step).astype(np.int64)
    idx = np.clip(starts[:, None] + offs[None, :], 0, h - 1)
    vals = row_prof[idx].sum(axis=1) / 9.0
    by = int(starts[int(np.argmax(vals))])

    # --- Bước 3: tinh chỉnh by ---
    bby, bbs = by, -1e18
    for by_c in range(max(0, by - 8), min(max_by, by + 9)):
        ii = np.clip(by_c + offs, 0, h - 1)
        s = float(row_prof[ii].mean())
        if s > bbs:
            bbs, bby = s, by_c
    by = int(max(0, min(bby, h - bw - 1)))

    return bx, by, bw


# ==============================================================================
# 2. ĐỌC 64 Ô BÀN CỜ
# ==============================================================================
def detect_board_cells(arr, bx, by, bw):
    h, w, _ = arr.shape
    cw = bw / 8.0
    lum = _luma(arr)
    chr_ = _chroma(arr)

    cells = []
    for r in range(8):
        for c in range(8):
            x0 = int(round(bx + c * cw))
            y0 = int(round(by + r * cw))
            pad = max(3, int(round(cw * 0.20)))
            xs0 = int(np.clip(x0 + pad, 0, w - 2))
            xs1 = int(np.clip(x0 + cw - pad, xs0 + 2, w))
            ys0 = int(np.clip(y0 + pad, 0, h - 2))
            ys1 = int(np.clip(y0 + cw - pad, ys0 + 2, h))
            block = arr[ys0:ys1, xs0:xs1]
            if block.size == 0:
                continue
            sl = lum[ys0:ys1, xs0:xs1]
            sc = chr_[ys0:ys1, xs0:xs1]
            cells.append({
                "r": r, "c": c,
                "rgb": block.reshape(-1, 3).mean(axis=0),
                "lum": float(sl.mean()),
                "chr": float(sc.mean()),
                "p10": float(np.percentile(sl, 8)),
                "p90": float(np.percentile(sl, 92)),
            })

    board = [[0] * 8 for _ in range(8)]
    conf = [[1.0] * 8 for _ in range(8)]
    uncertain = []
    board_mask = 0
    if len(cells) < 16:
        return board, board_mask, conf, uncertain

    dark = sorted(cells, key=lambda f: f["lum"])[: max(10, len(cells) // 5)]
    bg_rgb = np.median(np.array([f["rgb"] for f in dark]), axis=0)
    bg_l = float(0.299 * bg_rgb[0] + 0.587 * bg_rgb[1] + 0.114 * bg_rgb[2])
    bg_c = float(bg_rgb.max() - bg_rgb.min())

    for f in cells:
        d_l = f["lum"] - bg_l
        d_c = f["chr"] - bg_c
        dist = float(np.linalg.norm(f["rgb"] - bg_rgb))
        contrast = f["p90"] - f["p10"]
        f["score"] = (1.05 * max(0.0, d_l) + 0.85 * max(0.0, d_c)
                      + 0.55 * dist + 0.40 * max(0.0, contrast))
        f["dist"] = dist
        f["d_l"] = d_l
        f["d_c"] = d_c

    scores = np.array([f["score"] for f in cells], dtype=np.float64)
    spread = float(scores.max() - scores.min())

    if spread < 12.0:
        thr = float("inf")
    else:
        thr = float(np.clip(_otsu(scores),
                            scores.min() + spread * 0.32,
                            scores.min() + spread * 0.72))

    for f in cells:
        r, c = f["r"], f["c"]
        is_block = (f["score"] >= thr and
                    (f["d_l"] > 6.0 or f["d_c"] > 12.0 or f["dist"] > 20.0))
        if is_block:
            board[r][c] = 1
            board_mask |= (1 << (r * 8 + c))
        margin = abs(f["score"] - thr)
        cf = float(np.clip(0.55 + margin / 30.0, 0.55, 0.995))
        conf[r][c] = round(cf, 3)
        if cf < 0.72:
            uncertain.append({"row": r + 1, "col": c + 1, "confidence": round(cf, 3)})

    return board, board_mask, conf, uncertain


# ==============================================================================
# 3. ĐỌC 3 KHỐI CHỜ
# ==============================================================================
def _grid_fit_score(sub, col_occ, row_occ, rows, cols, ux, uy):
    purity, n = 0.0, 0
    for r in range(rows):
        for c in range(cols):
            cx0 = int(round(c * ux)); cx1 = int(round((c + 1) * ux))
            cy0 = int(round(r * uy)); cy1 = int(round((r + 1) * uy))
            px = max(2, int(round(ux * 0.24)))
            py = max(2, int(round(uy * 0.24)))
            cell = sub[max(0, cy0 + py):max(0, cy1 - py),
                       max(0, cx0 + px):max(0, cx1 - px)]
            if cell.size == 0:
                continue
            f = float(cell.mean())
            purity += 1.0 if (f >= 0.80 or f <= 0.20) else abs(f - 0.5) * 1.4
            n += 1
    purity = purity / float(max(1, n))

    bonus = 0.0
    if cols > 1:
        seams = [float(col_occ[max(0, int(round(c * ux)) - 1):
                               min(sub.shape[1], int(round(c * ux)) + 2)].mean())
                 for c in range(1, cols)]
        cents = [float(col_occ[min(sub.shape[1] - 1, int(round((c + 0.5) * ux)))])
                 for c in range(cols)]
        mc, ms = max(1e-6, float(np.mean(cents))), float(np.mean(seams))
        bonus += 1.6 * (mc - ms) / mc
    if rows > 1:
        seams = [float(row_occ[max(0, int(round(r * uy)) - 1):
                               min(sub.shape[0], int(round(r * uy)) + 2)].mean())
                 for r in range(1, rows)]
        cents = [float(row_occ[min(sub.shape[0] - 1, int(round((r + 0.5) * uy)))])
                 for r in range(rows)]
        mc, ms = max(1e-6, float(np.mean(cents))), float(np.mean(seams))
        bonus += 1.6 * (mc - ms) / mc
    return purity + bonus


def read_piece_grid(mask, bbox, unit_prior):
    x0, y0, x1, y1 = bbox
    bw, bh = x1 - x0 + 1, y1 - y0 + 1
    if bw < 9 or bh < 9:
        return None

    sub = mask[y0:y1 + 1, x0:x1 + 1].astype(np.float32)
    col_occ = sub.mean(axis=0)
    row_occ = sub.mean(axis=1)

    best = None
    u = max(4.0, max(bw, bh) / 5.0)
    u_hi = max(bw, bh) * 1.02
    while u <= u_hi:
        cols = int(round(bw / u))
        rows = int(round(bh / u))
        if 1 <= rows <= 5 and 1 <= cols <= 5:
            ux, uy = bw / cols, bh / rows
            if max(ux, uy) / max(1e-6, min(ux, uy)) <= 1.25:
                prior_err = abs(0.5 * (ux + uy) - unit_prior) / max(1.0, unit_prior)
                if prior_err <= 0.55:
                    fit = _grid_fit_score(sub, col_occ, row_occ, rows, cols, ux, uy)
                    total = fit - prior_err * 0.85
                    if best is None or total > best[0]:
                        best = (total, rows, cols, ux, uy)
        u += 1.0

    if best is None:
        cols = int(np.clip(round(bw / unit_prior), 1, 5))
        rows = int(np.clip(round(bh / unit_prior), 1, 5))
        best = (0.0, rows, cols, bw / cols, bh / rows)

    _, rows, cols, ux, uy = best
    mat = np.zeros((rows, cols), dtype=int)
    confs = []
    for r in range(rows):
        for c in range(cols):
            cx0 = int(round(c * ux)); cx1 = int(round((c + 1) * ux))
            cy0 = int(round(r * uy)); cy1 = int(round((r + 1) * uy))
            px = max(2, int(round(ux * 0.22)))
            py = max(2, int(round(uy * 0.22)))
            cell = sub[max(0, cy0 + py):max(0, cy1 - py),
                       max(0, cx0 + px):max(0, cx1 - px)]
            if cell.size == 0:
                continue
            f = float(cell.mean())
            mat[r, c] = 1 if f >= 0.50 else 0
            confs.append(float(np.clip(0.5 + abs(f - 0.5) * 0.95, 0.5, 0.995)))

    vr = np.any(mat == 1, axis=1)
    vc = np.any(mat == 1, axis=0)
    if not (np.any(vr) and np.any(vc)):
        return None

    return {
        "matrix": mat[vr][:, vc].tolist(),
        "confidence": round(float(np.mean(confs)) if confs else 0.5, 3),
        "bbox": (x0, y0, x1, y1),
    }


def detect_pieces(arr, y_top, cw):
    h, w, _ = arr.shape
    y_top = int(np.clip(y_top, 0, h - 20))
    region = arr[y_top:h]
    if region.shape[0] < 24:
        return [None, None, None], y_top

    lum = _luma(region)
    ch = _chroma(region)
    bg = np.median(region.reshape(-1, 3), axis=0)
    bg_l = float(0.299 * bg[0] + 0.587 * bg[1] + 0.114 * bg[2])
    bg_c = float(bg.max() - bg.min())

    d_l = lum - bg_l
    d_c = ch - bg_c
    dist = np.linalg.norm(region - bg, axis=2)

    energy = (np.maximum(0.0, d_l) * 0.85
              + np.maximum(0.0, d_c) * 0.95
              + dist * 0.30)
    energy[d_l < -5] *= 0.10          # bỏ bóng đổ
    energy[:2, :] = 0.0

    mx = float(energy.max())
    if mx < 22.0:
        return [None, None, None], y_top

    thr = float(np.clip(_otsu(energy[::3, ::3]), 12.0, mx * 0.62))
    mask = energy >= thr

    col_occ = np.convolve(mask.sum(axis=0).astype(np.float64),
                          np.ones(3) / 3.0, mode="same")
    if col_occ.max() <= 0:
        return [None, None, None], y_top

    runs = _group_runs(col_occ > max(2.5, col_occ.max() * 0.08),
                       min_gap=max(4, int(w * 0.012)))
    runs = [(a, b) for a, b in runs if b - a >= 10]

    band_w = w / 3.0
    slots = {}
    for a, b in runs:
        band = int(np.clip(((a + b) * 0.5) // band_w, 0, 2))
        if band in slots:
            a0, b0 = slots[band]
            slots[band] = (min(a0, a), max(b0, b))
        else:
            slots[band] = (a, b)

    unit_prior = cw * 0.62
    results = []
    for band in range(3):
        if band not in slots:
            results.append(None)
            continue
        a, b = slots[band]
        sub_mask = mask[:, a:b + 1]
        row_occ = np.convolve(sub_mask.sum(axis=1).astype(np.float64),
                              np.ones(3) / 3.0, mode="same")
        if row_occ.max() <= 0:
            results.append(None)
            continue
        rruns = _group_runs(row_occ > max(2.5, row_occ.max() * 0.08),
                            min_gap=max(4, int(h * 0.010)))
        if not rruns:
            results.append(None)
            continue
        ra, rb = max(rruns, key=lambda t: t[1] - t[0])
        results.append(read_piece_grid(mask, (a, ra, b, rb), unit_prior))

    return results, y_top


def detect_board_and_pieces(pil_img):
    img_hash = compute_image_sha256(pil_img)
    scaled, arr = normalize_image(pil_img, TARGET_W)
    h, w, _ = arr.shape

    bx, by, bw = locate_board(arr)
    cw = bw / 8.0

    board, board_mask, cell_confs, uncertain_cells = detect_board_cells(arr, bx, by, bw)

    dock_res, dock_y_top = detect_pieces(arr, by + bw, cw)
    if all(r is None for r in dock_res):
        alt, alt_y = detect_pieces(arr, int(h * 0.55), cw)
        if any(r is not None for r in alt):
            dock_res, dock_y_top = alt, alt_y

    pieces, slot_details, uncertain_slots = [], [], []

    for s in range(3):
        res = dock_res[s] if s < len(dock_res) else None
        if res:
            piece = {
                "slot": s + 1,
                "name": SLOT_NAMES[s],
                "matrix": res["matrix"],
                "confidence": res["confidence"],
                "placements": precompute_placements(res["matrix"]),
            }
            pieces.append(piece)
            x0, y0, x1, y1 = res["bbox"]
            slot_details.append({
                "slot": s + 1, "name": SLOT_NAMES[s], "detected": True,
                "matrix": res["matrix"], "confidence": res["confidence"],
                "slot_box": [x0, dock_y_top + y0, x1, dock_y_top + y1],
            })
            if res["confidence"] < 0.78:
                uncertain_slots.append({
                    "slot": s + 1, "name": SLOT_NAMES[s],
                    "reason": "Khối %s chỉ đạt %.1f%%" % (SLOT_NAMES[s], res["confidence"] * 100),
                })
        else:
            slot_details.append({
                "slot": s + 1, "name": SLOT_NAMES[s], "detected": False,
                "matrix": [], "confidence": 0.0, "slot_box": None,
            })
            uncertain_slots.append({
                "slot": s + 1, "name": SLOT_NAMES[s],
                "reason": "Không thấy khối ở ô %s" % SLOT_NAMES[s],
            })

    board_conf = float(np.mean(cell_confs))
    piece_conf = float(np.mean([p["confidence"] for p in pieces])) if pieces else 0.0

    meta = {
        "image_hash": img_hash,
        "bx": bx, "by": by, "bw": bw, "cw": cw,
        "img_w": w, "img_h": h,
        "scaled_img": scaled,
        "cell_confidences": cell_confs,
        "uncertain_cells": uncertain_cells,
        "uncertain_slots": uncertain_slots,
        "slot_details": slot_details,
        "overall_confidence": round(0.5 * board_conf + 0.5 * piece_conf, 3),
    }
    return board, board_mask, pieces, meta


# ==============================================================================
# 4. BỘ GIẢI
# ==============================================================================
def precompute_placements(mat):
    if not mat or not mat[0]:
        return []
    R, C = len(mat), len(mat[0])
    if R > 8 or C > 8:
        return []
    cells = sum(sum(row) for row in mat)
    if cells == 0:
        return []
    out = []
    for r0 in range(8 - R + 1):
        for c0 in range(8 - C + 1):
            pmask = 0
            for dr in range(R):
                for dc in range(C):
                    if mat[dr][dc] == 1:
                        pmask |= (1 << ((r0 + dr) * 8 + (c0 + dc)))
            out.append((r0, c0, pmask, cells))
    return out


def score_step(cells_placed, lines_cleared, combo_streak):
    pts = int(cells_placed)
    if lines_cleared <= 0:
        return pts, 0
    new_combo = combo_streak + 1
    clear = 10 * lines_cleared * (lines_cleared + 1) // 2 * new_combo
    return pts + clear, new_combo


def simulate(bmask, pmask, cells, combo_streak):
    nb = bmask | pmask
    clear_mask, rows_c, cols_c = 0, 0, 0
    for rm in ROW_MASKS:
        if (nb & rm) == rm:
            clear_mask |= rm
            rows_c += 1
    for cm in COL_MASKS:
        if (nb & cm) == cm:
            clear_mask |= cm
            cols_c += 1
    lines = rows_c + cols_c
    if clear_mask:
        nb &= ~clear_mask
    step_score, next_combo = score_step(cells, lines, combo_streak)
    if clear_mask and nb == 0:
        step_score += 50
    return nb, lines, rows_c, cols_c, step_score, next_combo


def heuristic(b):
    e = (~b) & FULL_MASK
    eh2 = e & (e >> 1) & NOT_RIGHT
    space_2x2 = (eh2 & (eh2 >> 8)).bit_count()
    eh3 = eh2 & (e >> 2) & 0x3F3F3F3F3F3F3F3F
    space_3x3 = (eh3 & (eh3 >> 8) & (eh3 >> 16)).bit_count()

    near = 0.0
    for rm in ROW_MASKS:
        cnt = (b & rm).bit_count()
        if cnt == 7:
            near += 8.0
        elif cnt == 6:
            near += 3.5
    for cm in COL_MASKS:
        cnt = (b & cm).bit_count()
        if cnt == 7:
            near += 8.0
        elif cnt == 6:
            near += 3.5

    filled = b.bit_count()
    return space_2x2 * 1.4 + space_3x3 * 3.0 + near + filled * 0.15


def _dfs(bm, order, idx, combo, real, rank, path, best, counter, deadline, ps):
    if counter[0] > 300000 or time.time() > deadline:
        return
    if idx >= len(order):
        if rank > best["rank"]:
            best["rank"] = rank
            best["real"] = real
            best["path"] = [dict(x) for x in path]
        return

    p = ps[order[idx]]
    moved = False
    for (r0, c0, pmask, cells) in p["placements"]:
        if bm & pmask:
            continue
        moved = True
        counter[0] += 1
        nb, lines, rc, cc, gain, ncombo = simulate(bm, pmask, cells, combo)
        hval = heuristic(nb)
        path.append({
            "slot": p["slot"], "name": p["name"], "matrix": p["matrix"],
            "r": r0, "c": c0, "gain": gain, "lines": lines,
            "rows_cleared": rc, "cols_cleared": cc,
        })
        _dfs(nb, order, idx + 1, ncombo, real + gain, real + gain + hval * 0.40,
             path, best, counter, deadline, ps)
        path.pop()

    if not moved:
        if rank > best["rank"]:
            best["rank"] = rank
            best["real"] = real
            best["path"] = [dict(x) for x in path]


def solve_board(board_mask, pieces):
    ps = []
    for p in pieces:
        m = p.get("matrix") or []
        if not m or not m[0]:
            continue
        pl = precompute_placements(m)
        if not pl:
            continue
        ps.append({"slot": p.get("slot", 0), "name": p.get("name", ""),
                   "matrix": m, "placements": pl})

    if not ps:
        return {"moves": [], "total_score": 0, "complete": True, "nodes": 0}

    n = len(ps)
    orders = list(itertools.permutations(range(n)))
    orders.sort(key=lambda o: sum(len(ps[i]["placements"]) for i in o))

    best = {"rank": -1e18, "real": -1, "path": []}
    counter = [0]
    deadline = time.time() + 3.0

    for order in orders:
        _dfs(board_mask, order, 0, 0, 0, 0.0, [], best, counter, deadline, ps)
        if counter[0] > 300000 or time.time() > deadline:
            break

    return {
        "moves": best["path"],
        "total_score": int(max(0, best["real"])),
        "complete": len(best["path"]) == n,
        "nodes": counter[0],
    }


# ==============================================================================
# 5. VẼ ẢNH KẾT QUẢ
# ==============================================================================
def render_solution(scaled, meta, solution):
    bx, by, bw = meta["bx"], meta["by"], meta["bw"]
    cw = bw / 8.0
    pad_l, pad_t = 48, 84
    W = scaled.width + pad_l + 12
    H = scaled.height + pad_t + 12

    canvas = Image.new("RGB", (W, H), (16, 18, 24))
    canvas.paste(scaled, (pad_l, pad_t))
    layer = Image.new("RGBA", (W, H), (0, 0, 0, 0))
    d = ImageDraw.Draw(layer)
    ox, oy = pad_l + bx, pad_t + by

    for k in range(9):
        x = ox + k * cw
        d.line([(x, oy), (x, oy + bw)], fill=(255, 255, 255, 55), width=1)
        y = oy + k * cw
        d.line([(ox, y), (ox + bw, y)], fill=(255, 255, 255, 55), width=1)

    if solution and solution["moves"]:
        for mv in solution["moves"]:
            col = SLOT_COLORS[(mv["slot"] - 1) % 3]
            mat = mv["matrix"]
            cells = [(rr, cc) for rr in range(len(mat))
                     for cc in range(len(mat[0])) if mat[rr][cc]]
            if not cells:
                continue
            xs = [mv["c"] + cc for _, cc in cells]
            ys = [mv["r"] + rr for rr, _ in cells]
            for (rr, cc) in cells:
                x0 = ox + (mv["c"] + cc) * cw
                y0 = oy + (mv["r"] + rr) * cw
                d.rectangle([x0 + 2, y0 + 2, x0 + cw - 2, y0 + cw - 2],
                            fill=col + (105,), outline=col + (255,), width=3)
            cxm = ox + (min(xs) + (max(xs) - min(xs) + 1) / 2.0) * cw
            cym = oy + (min(ys) + (max(ys) - min(ys) + 1) / 2.0) * cw
            fnum = get_font(max(14, int(cw * 0.55)))
            d.text((cxm, cym), "#%d" % mv["slot"], font=fnum,
                   fill=(255, 255, 255, 255), anchor="mm",
                   stroke_width=3, stroke_fill=(0, 0, 0, 235))

    canvas = Image.alpha_composite(canvas.convert("RGBA"), layer).convert("RGB")
    d2 = ImageDraw.Draw(canvas)

    f_lab = get_font(max(12, int(cw * 0.34)))
    for r in range(8):
        cy = oy + (r + 0.5) * cw
        d2.text((ox - 14, cy), str(r + 1), font=f_lab,
                fill=(200, 210, 225), anchor="rm")
    for c in range(8):
        cx = ox + (c + 0.5) * cw
        d2.text((cx, oy - 10), str(c + 1), font=f_lab,
                fill=(200, 210, 225), anchor="mb")

    f_h = get_font(20)
    f_b = get_font(15)
    d2.text((pad_l, 12), "BLOCK BLAST SOLVER", font=f_h, fill=(255, 255, 255))
    if solution and solution["moves"]:
        d2.text((pad_l, 44),
                "SCORE %d   |   %d/%d khối đặt được"
                % (solution["total_score"], len(solution["moves"]),
                   len([p for p in meta["slot_details"] if p["detected"]])),
                font=f_b, fill=(120, 235, 160))
    else:
        d2.text((pad_l, 44), "Không tìm được nước đi hợp lệ",
                font=f_b, fill=(255, 110, 110))

    lx = W - 132
    for i in range(3):
        col = SLOT_COLORS[i]
        y = 16 + i * 22
        d2.rectangle([lx, y, lx + 14, y + 14], fill=col)
        d2.text((lx + 20, y + 7), "#%d  %s" % (i + 1, SLOT_NAMES[i]),
                font=get_font(13), fill=(205, 212, 225), anchor="lm")

    d2.text((pad_l, H - 20),
            "Tin cậy: %.0f%%   |   Hàng/Cột tính từ 1"
            % (meta["overall_confidence"] * 100),
            font=get_font(12), fill=(140, 150, 165))

    return canvas


def png_bytes(img):
    buf = io.BytesIO()
    img.save(buf, format="PNG", optimize=True)
    return buf.getvalue()


def read_upload():
    f = request.files.get("image") or request.files.get("file")
    if f is not None:
        return Image.open(io.BytesIO(f.read()))
    raw = request.get_data()
    if not raw:
        return None
    return Image.open(io.BytesIO(raw))


# ==============================================================================
# 6. ROUTES
# ==============================================================================
@app.route("/", methods=["GET"])
@app.route("/health", methods=["GET"])
def index():
    return jsonify({
        "ok": True,
        "service": "Block Blast Solver",
        "endpoints": {
            "POST /solve?format=image": "Gửi ảnh (field: image) → trả ảnh PNG có đáp án",
            "POST /solve": "Gửi ảnh → trả JSON (board, pieces, solution)",
            "POST /inspect": "Gửi ảnh → trả JSON kiểm chứng nhận diện + preview base64",
            "GET /ui": "Giao diện sửa tay trên trình duyệt",
            "POST /ui/solve": "JSON {board, pieces} → đáp án",
        },
    })


@app.route("/solve", methods=["POST"])
def solve_endpoint():
    t0 = time.time()
    fmt = request.args.get("format", "json").lower()

    # Cho phép gửi JSON thuần (board + pieces) không cần ảnh
    if request.is_json:
        payload = request.get_json(silent=True) or {}
        if "board" in payload:
            bm = 0
            for r in range(min(8, len(payload["board"]))):
                for c in range(min(8, len(payload["board"][r]))):
                    if payload["board"][r][c]:
                        bm |= (1 << (r * 8 + c))
            pieces = [{"slot": i + 1, "name": SLOT_NAMES[i],
                       "matrix": p} for i, p in enumerate(payload.get("pieces", []))]
            sol = solve_board(bm, pieces)
            sol["elapsed_ms"] = int((time.time() - t0) * 1000)
            return jsonify({"ok": True, "solution": sol})

    img = read_upload()
    if img is None:
        return jsonify({"ok": False, "error": "Thiếu ảnh. Gửi field 'image'."}), 400

    try:
        board, board_mask, pieces, meta = detect_board_and_pieces(img)
        solution = solve_board(board_mask, pieces)
        solution["elapsed_ms"] = int((time.time() - t0) * 1000)

        if fmt == "image":
            out = render_solution(meta["scaled_img"], meta, solution)
            resp = Response(png_bytes(out), mimetype="image/png")
            resp.headers["X-Score"] = str(solution["total_score"])
            resp.headers["X-Confidence"] = "%.3f" % meta["overall_confidence"]
            return resp

        return jsonify({
            "ok": True,
            "board": board,
            "pieces": [{"slot": p["slot"], "name": p["name"],
                        "matrix": p["matrix"], "confidence": p["confidence"]}
                       for p in pieces],
            "solution": solution,
            "meta": {
                "bx": meta["bx"], "by": meta["by"], "bw": meta["bw"],
                "img_w": meta["img_w"], "img_h": meta["img_h"],
                "overall_confidence": meta["overall_confidence"],
                "uncertain_cells": meta["uncertain_cells"],
                "uncertain_slots": meta["uncertain_slots"],
            },
        })
    except Exception as exc:  # noqa: BLE001
        app.logger.exception("solve failed")
        return jsonify({"ok": False, "error": str(exc)}), 500


@app.route("/inspect", methods=["POST"])
def inspect_endpoint():
    img = read_upload()
    if img is None:
        return jsonify({"ok": False, "error": "Thiếu ảnh. Gửi field 'image'."}), 400

    try:
        board, board_mask, pieces, meta = detect_board_and_pieces(img)
        prev = meta["scaled_img"]
        buf = io.BytesIO()
        prev.save(buf, format="PNG", optimize=True)
        b64 = base64.b64encode(buf.getvalue()).decode()

        return jsonify({
            "ok": True,
            "image_hash": meta["image_hash"][:16],
            "image_w": meta["img_w"],
            "image_h": meta["img_h"],
            "board": {"bx": meta["bx"], "by": meta["by"],
                      "bw": meta["bw"], "cw": meta["cw"]},
            "board_matrix": board,
            "board_confidence": round(float(np.mean(meta["cell_confidences"])), 3),
            "cell_confidences": meta["cell_confidences"],
            "uncertain_cells": meta["uncertain_cells"],
            "pieces": meta["slot_details"],
            "uncertain_slots": meta["uncertain_slots"],
            "overall_confidence": meta["overall_confidence"],
            "preview": "data:image/png;base64," + b64,
        })
    except Exception as exc:  # noqa: BLE001
        app.logger.exception("inspect failed")
        return jsonify({"ok": False, "error": str(exc)}), 500


@app.route("/ui/solve", methods=["POST"])
def ui_solve():
    payload = request.get_json(silent=True) or {}
    bm = 0
    for r in range(min(8, len(payload.get("board", [])))):
        for c in range(min(8, len(payload["board"][r]))):
            if payload["board"][r][c]:
                bm |= (1 << (r * 8 + c))

    pieces = []
    for i, m in enumerate(payload.get("pieces", [])):
        if not m or not m[0]:
            continue
        pieces.append({"slot": i + 1, "name": SLOT_NAMES[i], "matrix": m})

    sol = solve_board(bm, pieces)
    try:
        with open(HISTORY_LOG_PATH, "a", encoding="utf-8") as fh:
            fh.write(json.dumps({"ts": time.time(), "board": payload.get("board"),
                                 "pieces": payload.get("pieces"),
                                 "score": sol["total_score"]}) + "\n")
    except Exception:
        pass
    return jsonify({"ok": True, "solution": sol})


@app.route("/history", methods=["GET"])
def history():
    rows = []
    try:
        with open(HISTORY_LOG_PATH, "r", encoding="utf-8") as fh:
            for line in fh.readlines()[-100:]:
                try:
                    rows.append(json.loads(line))
                except Exception:
                    continue
    except FileNotFoundError:
        pass
    return jsonify({"ok": True, "count": len(rows), "items": rows})


UI_HTML = r"""<!doctype html>
<html lang="vi"><head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>Block Blast – Sửa tay</title>
<style>
*{box-sizing:border-box}
body{margin:0;background:#101319;color:#e6eaf2;font-family:-apple-system,system-ui,sans-serif;padding:16px}
h1{font-size:18px;margin:0 0 12px}
button{background:#2f6fed;color:#fff;border:0;border-radius:8px;padding:10px 16px;font-size:14px;font-weight:600}
button:disabled{opacity:.45}
.row{display:flex;gap:12px;flex-wrap:wrap}
.panel{background:#181c25;border:1px solid #242a37;border-radius:12px;padding:14px}
#stage{position:relative;display:inline-block;max-width:100%}
#prev{max-width:100%;display:block;border-radius:8px}
#ov{position:absolute;inset:0}
.cell{position:absolute;border:1px solid rgba(255,255,255,.22);cursor:pointer}
.cell.on{background:rgba(255,82,82,.55)}
.pg{display:grid;grid-template-columns:repeat(5,26px);gap:2px;margin-bottom:10px}
.pg div{width:26px;height:26px;background:#232935;border-radius:4px;cursor:pointer}
.pg div.on{background:#4cd964}
#out{white-space:pre-wrap;font-size:13px;line-height:1.5;color:#9fe6b8}
</style></head><body>
<h1>Block Blast Solver – Sửa tay</h1>

<div class="panel" style="margin-bottom:12px">
  <input type="file" id="f" accept="image/*">
  <button id="go">Phân tích ảnh</button>
  <span id="msg" style="margin-left:8px;color:#8b95a8;font-size:13px"></span>
</div>

<div class="row">
  <div class="panel">
    <div id="stage"><img id="prev" alt=""><div id="ov"></div></div>
  </div>

  <div class="panel">
    <b style="font-size:13px">3 khối (ô xanh = có gạch)</b>
    <div class="row" id="pgs" style="margin-top:8px"></div>
    <button id="solve">Tính đáp án</button>
    <div id="out" style="margin-top:10px"></div>
  </div>
</div>

<script>
const $=s=>document.querySelector(s);
let board=Array.from({length:8},()=>Array(8).fill(0));
let pieces=[0,1,2].map(()=>Array.from({length:5},()=>Array(5).fill(0)));
let geo=null;

function buildOverlay(){
  const ov=$('#ov'); ov.innerHTML='';
  if(!geo) return;
  const img=$('#prev');
  const W=img.clientWidth, H=img.clientHeight;
  if(!W) return;
  const {bx,by,bw}=geo;
  const cw=bw/8;
  for(let r=0;r<8;r++)for(let c=0;c<8;c++){
    const d=document.createElement('div');
    d.className='cell'+(board[r][c]?' on':'');
    d.style.left=(bx+c*cw)/img.naturalWidth*W+'px';
    d.style.top=(by+r*cw)/img.naturalHeight*H+'px';
    d.style.width=cw/img.naturalWidth*W+'px';
    d.style.height=cw/img.naturalHeight*H+'px';
    d.onclick=()=>{board[r][c]^=1;buildOverlay();};
    ov.appendChild(d);
  }
}

function buildPieces(){
  const wrap=$('#pgs'); wrap.innerHTML='';
  pieces.forEach((m,pi)=>{
    const g=document.createElement('div'); g.className='pg';
    for(let r=0;r<5;r++)for(let c=0;c<5;c++){
      const d=document.createElement('div');
      if(m[r][c]) d.className='on';
      d.onclick=()=>{m[r][c]^=1;buildPieces();};
      g.appendChild(d);
    }
    wrap.appendChild(g);
  });
}

function trim(m){
  let r0=5,r1=-1,c0=5,c1=-1;
  for(let r=0;r<5;r++)for(let c=0;c<5;c++) if(m[r][c]){
    r0=Math.min(r0,r);r1=Math.max(r1,r);c0=Math.min(c0,c);c1=Math.max(c1,c);
  }
  if(r1<0) return null;
  return m.slice(r0,r1+1).map(row=>row.slice(c0,c1+1));
}

$('#go').onclick=async()=>{
  const file=$('#f').files[0];
  if(!file){ $('#msg').textContent='Chọn ảnh trước đã.'; return; }
  $('#go').disabled=true; $('#msg').textContent='Đang phân tích...';
  try{
    const fd=new FormData(); fd.append('image',file);
    const r=await fetch('/inspect',{method:'POST',body:fd});
    const j=await r.json();
    if(!j.ok) throw new Error(j.error||'lỗi');
    board=j.board_matrix.map(x=>x.slice());
    geo=j.board;
    const img=$('#prev');
    img.onload=buildOverlay;
    img.src=j.preview;
    pieces=[0,1,2].map(i=>{
      const p=j.pieces[i];
      const m=Array.from({length:5},()=>Array(5).fill(0));
      if(p&&p.detected) p.matrix.forEach((row,r)=>row.forEach((v,c)=>{if(r<5&&c<5)m[r][c]=v;}));
      return m;
    });
    buildPieces();
    $('#msg').textContent='Tin cậy '+Math.round(j.overall_confidence*100)+'% – kiểm tra rồi bấm Tính đáp án';
  }catch(e){ $('#msg').textContent='Lỗi: '+e.message; }
  $('#go').disabled=false;
};

$('#solve').onclick=async()=>{
  $('#out').textContent='Đang giải...';
  const ps=pieces.map(trim).filter(Boolean);
  const r=await fetch('/ui/solve',{method:'POST',headers:{'Content-Type':'application/json'},
    body:JSON.stringify({board,pieces:ps})});
  const j=await r.json();
  if(!j.ok){ $('#out').textContent='Lỗi'; return; }
  const s=j.solution;
  let txt='SCORE: '+s.total_score+'\n';
  s.moves.forEach(m=>{
    txt+='#'+m.slot+' ('+m.name+') → Hàng '+(m.r+1)+', Cột '+(m.c+1)
        +'  (+'+m.gain+' điểm)\n';
  });
  if(!s.moves.length) txt='Không có nước đi hợp lệ.';
  $('#out').textContent=txt;
};

buildPieces();
</script>
</body></html>"""


@app.route("/ui", methods=["GET"])
def ui_page():
    return Response(UI_HTML, mimetype="text/html; charset=utf-8")


if __name__ == "__main__":
    port = int(os.environ.get("PORT", 10000))
    app.run(host="0.0.0.0", port=port, debug=False)