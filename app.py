import io
import os
import sys
import json
import time
import uuid
import base64
import hashlib
import itertools
from collections import deque
import numpy as np
from PIL import Image, ImageDraw
from flask import Flask, request, send_file, jsonify, Response

app = Flask(__name__)

# ==============================================================================
# CẤU HÌNH LƯU TRỮ PHIÊN KIỂM CHỨNG & LỊCH SỬ GIẢI
# ==============================================================================
DATA_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "solver_data")
os.makedirs(DATA_DIR, exist_ok=True)
HISTORY_LOG_PATH = os.path.join(DATA_DIR, "solver_history.jsonl")
SESSIONS_CACHE = {}

# ==============================================================================
# HẰNG SỐ BITBOARD 64-BIT
# ==============================================================================
FULL_MASK = 0xFFFFFFFFFFFFFFFF
NOT_RIGHT = 0x7F7F7F7F7F7F7F7F
NOT_LEFT = 0xFEFEFEFEFEFEFEFE
ROW_MASKS = [0xFF << (r * 8) for r in range(8)]
COL_MASKS = [0x0101010101010101 << c for c in range(8)]


# ==============================================================================
# 1. HỆ THỐNG XỬ LÝ ẢNH & NHẬN DIỆN ĐỘ CHÍNH XÁC CAO
# ==============================================================================
def compute_image_sha256(pil_img):
    raw_bytes = pil_img.convert("RGB").tobytes()
    return hashlib.sha256(raw_bytes).hexdigest()[:16]


def normalize_image(pil_img, target_w=720):
    rgb_img = pil_img.convert("RGB")
    if rgb_img.width == target_w:
        scaled = rgb_img
    else:
        scale = target_w / float(rgb_img.width)
        target_h = int(round(rgb_img.height * scale))
        scaled = rgb_img.resize((target_w, target_h), Image.Resampling.LANCZOS)
    arr = np.asarray(scaled, dtype=np.float32)
    return scaled, arr


def rgb_to_lum_chroma(arr):
    lum = 0.299 * arr[:, :, 0] + 0.587 * arr[:, :, 1] + 0.114 * arr[:, :, 2]
    chroma = arr.max(axis=2) - arr.min(axis=2)
    return lum, chroma


def auto_locate_board(rgb_arr):
    h, w, _ = rgb_arr.shape
    lum, _ = rgb_to_lum_chroma(rgb_arr)

    vert_grad = np.abs(np.diff(lum, axis=0, prepend=lum[:1, :]))
    horiz_grad = np.abs(np.diff(lum, axis=1, prepend=lum[:, :1]))

    mid_y0, mid_y1 = int(h * 0.18), int(h * 0.65)
    col_edge_profile = horiz_grad[mid_y0:mid_y1, :].mean(axis=0)

    default_bw = int(round(w * 0.908))
    default_bx = int(round((w - default_bw) / 2.0))

    min_m = max(6, int(w * 0.025))
    max_m = int(w * 0.11)
    best_bx = default_bx
    best_bw = default_bw
    best_x_score = -1e9

    for m in range(min_m, max_m):
        cand_bw = w - 2 * m
        if cand_bw < int(w * 0.78):
            continue
        cw_cand = cand_bw / 8.0
        l_peak = col_edge_profile[max(0, m - 2) : min(w, m + 3)].max()
        r_idx = m + cand_bw
        r_peak = col_edge_profile[max(0, r_idx - 2) : min(w, r_idx + 3)].max()

        internal_v = 0.0
        for k in range(1, 8):
            gx = int(round(m + k * cw_cand))
            if 2 <= gx < w - 2:
                internal_v += col_edge_profile[gx - 2 : gx + 3].max()

        reg_penalty = abs(m - default_bx) * 0.45
        score_x = l_peak + r_peak + (internal_v / 7.0) * 0.85 - reg_penalty
        if score_x > best_x_score:
            best_x_score = score_x
            best_bx = m
            best_bw = cand_bw

    row_edge_profile = vert_grad[:, best_bx : best_bx + best_bw].mean(axis=1)

    y_min = max(8, int(h * 0.11))
    y_max = min(int(h * 0.36), h - best_bw - 20)
    best_by = int(round(h * 0.228))
    best_y_score = -1e9
    cw_est = best_bw / 8.0

    for y in range(y_min, max(y_min + 1, y_max)):
        bot_y = y + best_bw
        if bot_y >= h - 4:
            continue
        top_peak = row_edge_profile[max(0, y - 3) : min(h, y + 4)].max()
        bot_peak = row_edge_profile[max(0, bot_y - 3) : min(h, bot_y + 4)].max()

        internal_h = 0.0
        for k in range(1, 8):
            gy = int(round(y + k * cw_est))
            if 2 <= gy < h - 2:
                internal_h += row_edge_profile[gy - 2 : gy + 3].max()

        score_y = top_peak + bot_peak * 0.92 + (internal_h / 7.0) * 0.85
        if score_y > best_y_score:
            best_y_score = score_y
            best_by = y

    return best_bx, best_by, best_bw


def detect_board_cells(rgb_arr, bx, by, bw):
    cw = bw / 8.0
    lum_full, chroma_full = rgb_to_lum_chroma(rgb_arr)

    seam_samples = []
    for k in range(1, 8):
        gy = int(round(by + k * cw))
        if 0 <= gy < rgb_arr.shape[0]:
            seam_samples.append(rgb_arr[gy, bx : bx + bw, :])
    if seam_samples:
        seam_rgb = np.median(np.concatenate(seam_samples, axis=0), axis=0)
    else:
        seam_rgb = np.array([20.0, 26.0, 52.0], dtype=np.float32)

    cell_feats = []
    for r in range(8):
        for c in range(8):
            x0 = int(round(bx + c * cw))
            y0 = int(round(by + r * cw))
            x1 = int(round(bx + (c + 1) * cw))
            y1 = int(round(by + (r + 1) * cw))

            cell_w = max(8, x1 - x0)
            cell_h = max(8, y1 - y0)

            pad_x = max(2, int(round(cell_w * 0.14)))
            pad_y = max(2, int(round(cell_h * 0.14)))
            core_rgb = rgb_arr[y0 + pad_y : y1 - pad_y, x0 + pad_x : x1 - pad_x]
            core_lum = lum_full[y0 + pad_y : y1 - pad_y, x0 + pad_x : x1 - pad_x]
            core_chroma = chroma_full[y0 + pad_y : y1 - pad_y, x0 + pad_x : x1 - pad_x]

            in_x = max(3, int(round(cell_w * 0.28)))
            in_y = max(3, int(round(cell_h * 0.28)))
            center_rgb = rgb_arr[y0 + in_y : y1 - in_y, x0 + in_x : x1 - in_x]
            mean_rgb = center_rgb.mean(axis=(0, 1)) if center_rgb.size > 0 else core_rgb.mean(axis=(0, 1))

            std_lum = float(core_lum.std()) if core_lum.size > 0 else 0.0
            mean_lum = float(core_lum.mean()) if core_lum.size > 0 else 0.0
            mean_chroma = float(core_chroma.mean()) if core_chroma.size > 0 else 0.0

            b_strip = max(2, int(round(core_lum.shape[0] * 0.22)))
            top_l = float(core_lum[:b_strip, :].mean())
            bot_l = float(core_lum[-b_strip:, :].mean())
            left_l = float(core_lum[:, :b_strip].mean())
            right_l = float(core_lum[:, -b_strip:].mean())

            abs_bevel = abs(top_l - bot_l) + abs(left_l - right_l)
            signed_bevel = max(0.0, top_l - bot_l) + max(0.0, left_l - right_l)
            flatness = std_lum * 1.1 + abs_bevel * 1.15 + mean_chroma * 0.45

            cell_feats.append({
                "r": r,
                "c": c,
                "rgb": mean_rgb,
                "lum": mean_lum,
                "chroma": mean_chroma,
                "std": std_lum,
                "bevel": abs_bevel,
                "signed_bevel": signed_bevel,
                "flatness": flatness,
            })

    sorted_flat = sorted(cell_feats, key=lambda x: x["flatness"])
    darkest_flat = [
        x for x in sorted_flat[:16]
        if x["bevel"] < 14.0 and x["std"] < 10.0 and x["chroma"] < 48.0
    ]
    if darkest_flat:
        anchor = darkest_flat[0]["rgb"]
        cluster = [
            x["rgb"] for x in darkest_flat
            if float(np.linalg.norm(x["rgb"] - anchor)) < 18.0
        ]
        bg_rgb = np.median(np.array(cluster), axis=0)
    else:
        bg_rgb = seam_rgb * 1.12

    bg_lum = float(0.299 * bg_rgb[0] + 0.587 * bg_rgb[1] + 0.114 * bg_rgb[2])
    bg_chroma = float(bg_rgb.max() - bg_rgb.min())

    scores = []
    for f in cell_feats:
        color_dist = float(np.linalg.norm(f["rgb"] - bg_rgb))
        chroma_gain = max(0.0, f["chroma"] - bg_chroma)
        lum_gain = max(0.0, f["lum"] - bg_lum)

        evidence = (
            color_dist * 1.05
            + chroma_gain * 0.65
            + lum_gain * 0.35
            + f["bevel"] * 1.10
            + f["signed_bevel"] * 0.55
            + f["std"] * 0.85
        )
        f["evidence"] = evidence
        scores.append(evidence)

    arr_s = np.asarray(scores, dtype=np.float32)
    s_min, s_max = float(arr_s.min()), float(arr_s.max())

    if s_max < 28.0:
        decision_thresh = 32.0
    elif s_min > 48.0:
        decision_thresh = 30.0
    else:
        best_t = 34.0
        best_var = -1.0
        for step in range(1, 64):
            t = s_min + (s_max - s_min) * (step / 64.0)
            w0 = arr_s[arr_s <= t]
            w1 = arr_s[arr_s > t]
            if len(w0) == 0 or len(w1) == 0:
                continue
            var_b = float(len(w0)) * float(len(w1)) * ((float(w0.mean()) - float(w1.mean())) ** 2)
            if var_b > best_var:
                best_var = var_b
                best_t = t
        decision_thresh = float(np.clip(best_t, 26.0, 56.0))

    board = [[0] * 8 for _ in range(8)]
    board_mask = 0
    cell_confidences = [[1.0] * 8 for _ in range(8)]
    uncertain_cells = []

    for f in cell_feats:
        r, c = f["r"], f["c"]
        ev = f["evidence"]
        is_block = (ev >= decision_thresh) and (
            f["bevel"] >= 9.0 or f["chroma"] >= bg_chroma + 16.0 or float(np.linalg.norm(f["rgb"] - bg_rgb)) >= 22.0
        )
        if is_block:
            board[r][c] = 1
            board_mask |= (1 << (r * 8 + c))

        margin = abs(ev - decision_thresh)
        conf = float(np.clip(0.50 + (margin / 28.0) * 0.50, 0.50, 0.999))
        cell_confidences[r][c] = round(conf, 3)
        if conf < 0.72:
            uncertain_cells.append({"row": r + 1, "col": c + 1, "confidence": round(conf, 3)})

    return board, board_mask, cell_confidences, uncertain_cells


def clean_connected_mask(raw_mask):
    sh, sw = raw_mask.shape
    if sh < 5 or sw < 5:
        return raw_mask

    padded = np.pad(raw_mask.astype(np.uint8), 1, mode="constant", constant_values=0)
    n_sum = (
        padded[0:sh, 0:sw] + padded[0:sh, 1 : sw + 1] + padded[0:sh, 2 : sw + 2]
        + padded[1 : sh + 1, 0:sw] + padded[1 : sh + 1, 1 : sw + 1] + padded[1 : sh + 1, 2 : sw + 2]
        + padded[2 : sh + 2, 0:sw] + padded[2 : sh + 2, 1 : sw + 1] + padded[2 : sh + 2, 2 : sw + 2]
    )
    core = raw_mask & (n_sum >= 4)
    if not np.any(core):
        return np.zeros_like(raw_mask, dtype=bool)

    p_core = np.pad(core.astype(np.uint8), 1, mode="constant", constant_values=0)
    dilated = (
        p_core[0:sh, 1 : sw + 1]
        + p_core[2 : sh + 2, 1 : sw + 1]
        + p_core[1 : sh + 1, 0:sw]
        + p_core[1 : sh + 1, 2 : sw + 2]
        + p_core[1 : sh + 1, 1 : sw + 1]
    ) > 0
    morphed = raw_mask & dilated

    visited = np.zeros((sh, sw), dtype=bool)
    comps = []
    ys, xs = np.where(morphed)
    for y0, x0 in zip(ys, xs):
        if visited[y0, x0]:
            continue
        q = deque([(int(y0), int(x0))])
        visited[y0, x0] = True
        comp = []
        while q:
            cy, cx = q.popleft()
            comp.append((cy, cx))
            for ny in range(max(0, cy - 1), min(sh, cy + 2)):
                for nx in range(max(0, cx - 1), min(sw, cx + 2)):
                    if not visited[ny, nx] and morphed[ny, nx]:
                        visited[ny, nx] = True
                        q.append((ny, nx))
        comps.append(comp)

    if not comps:
        return np.zeros_like(raw_mask, dtype=bool)

    comps.sort(key=len, reverse=True)
    max_area = len(comps[0])
    if max_area < 24:
        return np.zeros_like(raw_mask, dtype=bool)

    m_ys = [p[0] for p in comps[0]]
    m_xs = [p[1] for p in comps[0]]
    my0, my1 = min(m_ys), max(m_ys)
    mx0, mx1 = min(m_xs), max(m_xs)

    cleaned = np.zeros((sh, sw), dtype=bool)
    for comp in comps:
        if len(comp) < max(16, int(max_area * 0.07)):
            continue
        c_ys = [p[0] for p in comp]
        c_xs = [p[1] for p in comp]
        cy0, cy1 = min(c_ys), max(c_ys)
        cx0, cx1 = min(c_xs), max(c_xs)
        dist_y = max(0, my0 - cy1, cy0 - my1)
        dist_x = max(0, mx0 - cx1, cx0 - mx1)
        if dist_y <= 14 and dist_x <= 14:
            for py, px in comp:
                cleaned[py, px] = True

    return cleaned


def solve_piece_grid_dimensions(bw_p, bh_p, unit_base, roi_lum, roi_chroma, roi_mask):
    aspect = float(bw_p) / max(1.0, float(bh_p))

    col_lum = (roi_lum * roi_mask).sum(axis=0) / np.maximum(1.0, roi_mask.sum(axis=0))
    row_lum = (roi_lum * roi_mask).sum(axis=1) / np.maximum(1.0, roi_mask.sum(axis=1))
    col_grad = np.abs(np.diff(col_lum, prepend=col_lum[:1]))
    row_grad = np.abs(np.diff(row_lum, prepend=row_lum[:1]))

    candidates = []

    for r in range(1, 6):
        for c in range(1, 6):
            ux = float(bw_p) / float(c)
            uy = float(bh_p) / float(r)
            u_mean = 0.5 * (ux + uy)

            if u_mean < unit_base * 0.55 or u_mean > unit_base * 1.45:
                continue

            sq_err = abs(ux - uy) / max(ux, uy)
            model_aspect = float(c) / float(r)
            aspect_err = abs(aspect - model_aspect) / max(aspect, model_aspect)

            if sq_err > 0.24:
                continue

            max_dim_cells = max(r, c)
            if max_dim_cells == 5:
                expected_u = float(np.clip(u_mean, unit_base * 0.80, unit_base * 0.98))
                scale_err = abs(u_mean - expected_u) / unit_base
            elif max_dim_cells == 4:
                expected_u = float(np.clip(u_mean, unit_base * 0.91, unit_base * 1.05))
                scale_err = abs(u_mean - expected_u) / unit_base
            else:
                expected_u = float(np.clip(u_mean, unit_base * 0.92, unit_base * 1.10))
                scale_err = abs(u_mean - expected_u) / unit_base

            seam_score = 0.0
            if c > 1 and bw_p >= c * 10:
                peaks = []
                mids = []
                for k in range(1, c):
                    sx = int(round(k * ux))
                    rad = max(2, int(round(ux * 0.12)))
                    peaks.append(float(col_grad[max(0, sx - rad) : min(bw_p, sx + rad + 1)].max()))
                for k in range(c):
                    mx = int(round((k + 0.5) * ux))
                    rad = max(2, int(round(ux * 0.15)))
                    mids.append(float(col_grad[max(0, mx - rad) : min(bw_p, mx + rad + 1)].mean()))
                mean_grad = float(col_grad.mean()) + 1.0
                seam_score += (float(np.mean(peaks)) - float(np.mean(mids))) / mean_grad

            if r > 1 and bh_p >= r * 10:
                peaks = []
                mids = []
                for k in range(1, r):
                    sy = int(round(k * uy))
                    rad = max(2, int(round(uy * 0.12)))
                    peaks.append(float(row_grad[max(0, sy - rad) : min(bh_p, sy + rad + 1)].max()))
                for k in range(r):
                    my = int(round((k + 0.5) * uy))
                    rad = max(2, int(round(uy * 0.15)))
                    mids.append(float(row_grad[max(0, my - rad) : min(bh_p, my + rad + 1)].mean()))
                mean_grad = float(row_grad.mean()) + 1.0
                seam_score += (float(np.mean(peaks)) - float(np.mean(mids))) / mean_grad

            seam_bonus = float(np.clip(seam_score * 0.045, -0.06, 0.12))
            total_cost = sq_err * 3.4 + aspect_err * 2.4 + scale_err * 1.5 - seam_bonus
            candidates.append((total_cost, r, c))

    if not candidates:
        r_fallback = int(np.clip(round(bh_p / unit_base), 1, 5))
        c_fallback = int(np.clip(round(bw_p / unit_base), 1, 5))
        return r_fallback, c_fallback, 0.65

    candidates.sort(key=lambda x: x[0])
    best_cost, best_r, best_c = candidates[0]
    if len(candidates) > 1:
        second_cost = candidates[1][0]
        margin = max(0.0, second_cost - best_cost)
        dim_conf = float(np.clip(0.75 + margin * 0.85, 0.70, 0.999))
    else:
        dim_conf = 0.95

    return best_r, best_c, dim_conf


def extract_piece_from_slot(slot_arr, dock_bg_global, unit_base):
    sh, sw, _ = slot_arr.shape
    if sh < 16 or sw < 16:
        return None

    border_pixels = np.concatenate([
        slot_arr[0:4, :, :].reshape(-1, 3),
        slot_arr[-4:, :, :].reshape(-1, 3),
        slot_arr[:, 0:4, :].reshape(-1, 3),
        slot_arr[:, -4:, :].reshape(-1, 3),
    ], axis=0)
    local_bg = np.median(border_pixels, axis=0)
    dock_bg = 0.70 * local_bg + 0.30 * dock_bg_global

    bg_lum = float(0.299 * dock_bg[0] + 0.587 * dock_bg[1] + 0.114 * dock_bg[2])
    bg_chroma = float(dock_bg.max() - dock_bg.min())

    lum, chroma = rgb_to_lum_chroma(slot_arr)
    color_diff = np.linalg.norm(slot_arr - dock_bg, axis=2)

    lum_gain = np.maximum(0.0, lum - bg_lum)
    chroma_gain = np.maximum(0.0, chroma - bg_chroma)

    is_shadow = (lum < bg_lum + 7.5) & (chroma_gain < 20.0) & (color_diff < 52.0)

    fg_energy = color_diff * 0.85 + lum_gain * 0.65 + chroma_gain * 0.95
    fg_energy[is_shadow] = 0.0
    fg_energy[:3, :] = 0.0
    fg_energy[-3:, :] = 0.0
    fg_energy[:, :2] = 0.0
    fg_energy[:, -2:] = 0.0

    max_energy = float(fg_energy.max())
    if max_energy < 26.0:
        return None

    active = fg_energy[fg_energy > 14.0]
    if active.size < 40:
        return None

    vmin, vmax = float(active.min()), float(active.max())
    best_t = (vmin + vmax) * 0.45
    best_var = -1.0
    for step in range(1, 48):
        t = vmin + (vmax - vmin) * (step / 48.0)
        w0 = active[active <= t]
        w1 = active[active > t]
        if len(w0) == 0 or len(w1) == 0:
            continue
        var_b = float(len(w0)) * float(len(w1)) * ((float(w0.mean()) - float(w1.mean())) ** 2)
        if var_b > best_var:
            best_var = var_b
            best_t = t

    thresh = max(25.0, min(float(best_t), max_energy * 0.52))
    raw_mask = fg_energy >= thresh
    mask = clean_connected_mask(raw_mask)

    ys, xs = np.where(mask)
    if len(xs) < 40:
        return None

    col_counts = mask.sum(axis=0)
    row_counts = mask.sum(axis=1)
    vx = np.where(col_counts >= max(3, int(col_counts.max() * 0.14)))[0]
    vy = np.where(row_counts >= max(3, int(row_counts.max() * 0.14)))[0]
    if len(vx) == 0 or len(vy) == 0:
        return None

    min_x, max_x = int(vx[0]), int(vx[-1])
    min_y, max_y = int(vy[0]), int(vy[-1])

    while max_y - min_y > 14:
        bot_mask = mask[max_y, min_x : max_x + 1]
        if np.any(bot_mask):
            bot_lum = float(lum[max_y, min_x : max_x + 1][bot_mask].mean())
            bot_chr = float(chroma[max_y, min_x : max_x + 1][bot_mask].mean())
            core_region_mask = mask[min_y:max_y, min_x : max_x + 1]
            core_lum = float(lum[min_y:max_y, min_x : max_x + 1][core_region_mask].mean())
            if (bot_lum < bg_lum + 12.0 and bot_lum < core_lum * 0.58) or (
                bot_chr < bg_chroma + 14.0 and bot_lum < core_lum * 0.62
            ):
                mask[max_y, :] = False
                max_y -= 1
                continue
        break

    bw_p = max_x - min_x + 1
    bh_p = max_y - min_y + 1
    if bw_p < 10 or bh_p < 10:
        return None

    roi_lum = lum[min_y : max_y + 1, min_x : max_x + 1]
    roi_chroma = chroma[min_y : max_y + 1, min_x : max_x + 1]
    roi_mask = mask[min_y : max_y + 1, min_x : max_x + 1]

    rows, cols, dim_conf = solve_piece_grid_dimensions(
        bw_p, bh_p, unit_base, roi_lum, roi_chroma, roi_mask
    )

    fg_pixels = slot_arr[mask]
    piece_rgb = np.median(fg_pixels, axis=0)
    fg_bg_dist = max(20.0, float(np.linalg.norm(piece_rgb - dock_bg)))

    mat = np.zeros((rows, cols), dtype=int)
    sub_confs = []
    sub_w = bw_p / float(cols)
    sub_h = bh_p / float(rows)

    for r in range(rows):
        for c in range(cols):
            cx0 = int(round(min_x + (c + 0.18) * sub_w))
            cx1 = int(round(min_x + (c + 0.82) * sub_w))
            cy0 = int(round(min_y + (r + 0.18) * sub_h))
            cy1 = int(round(min_y + (r + 0.82) * sub_h))

            cx0, cx1 = max(0, cx0), min(sw, max(cx0 + 1, cx1))
            cy0, cy1 = max(0, cy0), min(sh, max(cy0 + 1, cy1))

            sub_m = mask[cy0:cy1, cx0:cx1]
            sub_rgb = slot_arr[cy0:cy1, cx0:cx1]
            if sub_m.size == 0:
                sub_confs.append(0.5)
                continue

            fill_ratio = float(sub_m.mean())
            mean_c_rgb = sub_rgb.mean(axis=(0, 1))
            d_bg = float(np.linalg.norm(mean_c_rgb - dock_bg))
            d_fg = float(np.linalg.norm(mean_c_rgb - piece_rgb))
            color_affinity = d_bg / max(1.0, d_bg + d_fg)

            occupancy_score = fill_ratio * 0.65 + color_affinity * 0.35
            if fill_ratio >= 0.40 or (fill_ratio >= 0.24 and color_affinity >= 0.58 and d_bg >= fg_bg_dist * 0.45):
                mat[r, c] = 1
                cell_conf = float(np.clip(0.55 + (occupancy_score - 0.38) * 1.1, 0.55, 0.999))
            else:
                mat[r, c] = 0
                cell_conf = float(np.clip(0.55 + (0.42 - fill_ratio) * 1.1, 0.55, 0.999))
            sub_confs.append(cell_conf)

    valid_r = np.any(mat == 1, axis=1)
    valid_c = np.any(mat == 1, axis=0)
    if not (np.any(valid_r) and np.any(valid_c)):
        return None

    trimmed_mat = mat[valid_r][:, valid_c].tolist()
    overall_conf = round(float(min(dim_conf, np.mean(sub_confs) if sub_confs else 0.8)), 3)

    return {
        "matrix": trimmed_mat,
        "confidence": overall_conf,
        "bbox": (min_x, min_y, max_x, max_y),
        "raw_dims": (rows, cols),
    }


def detect_board_and_pieces(pil_img):
    img_hash = compute_image_sha256(pil_img)
    scaled_img, arr = normalize_image(pil_img, target_w=720)
    target_h, target_w, _ = arr.shape

    bx, by, bw = auto_locate_board(arr)
    cw = bw / 8.0

    board, board_mask, cell_confs, uncertain_cells = detect_board_cells(arr, bx, by, bw)

    dock_y0 = min(target_h - 60, int(round(by + bw + max(cw * 0.42, target_h * 0.035))))
    safe_bottom = target_h - max(18, int(round(target_h * 0.038)))
    dock_y1 = min(safe_bottom, int(round(dock_y0 + max(cw * 3.15, target_h * 0.195))))
    if dock_y1 <= dock_y0 + 32:
        dock_y1 = min(target_h - 10, dock_y0 + 96)

    dock_x0 = int(round(target_w * 0.035))
    slot_w = int(round((target_w * 0.93) / 3.0))
    unit_base = cw * 0.472

    dock_strip = arr[dock_y0:dock_y1, dock_x0 : min(target_w, dock_x0 + 3 * slot_w)]
    dock_borders = np.concatenate([
        dock_strip[0:4, :, :].reshape(-1, 3),
        dock_strip[-4:, :, :].reshape(-1, 3),
    ], axis=0)
    dock_bg_global = np.median(dock_borders, axis=0)

    pieces = []
    slot_names = ["TRÁI", "GIỮA", "PHẢI"]
    uncertain_slots = []
    slot_details = []

    for s in range(3):
        sx0 = dock_x0 + s * slot_w + 4
        sx1 = min(target_w - 2, dock_x0 + (s + 1) * slot_w - 4)
        slot_arr = arr[dock_y0:dock_y1, sx0:sx1]

        res = extract_piece_from_slot(slot_arr, dock_bg_global, unit_base)
        if res is not None:
            mat = res["matrix"]
            conf = res["confidence"]
            p_obj = {
                "slot": s + 1,
                "name": slot_names[s],
                "matrix": mat,
                "confidence": conf,
                "placements": precompute_placements(mat),
            }
            pieces.append(p_obj)
            slot_details.append({
                "slot": s + 1,
                "name": slot_names[s],
                "detected": True,
                "matrix": mat,
                "confidence": conf,
                "slot_box": (sx0, dock_y0, sx1, dock_y1),
                "piece_bbox": res["bbox"],
            })
            if conf < 0.78:
                uncertain_slots.append({
                    "slot": s + 1,
                    "name": slot_names[s],
                    "reason": f"Độ tin cậy khối {slot_names[s]} đạt {conf*100:.1f}% (< 78%)",
                })
        else:
            slot_details.append({
                "slot": s + 1,
                "name": slot_names[s],
                "detected": False,
                "matrix": [],
                "confidence": 0.0,
                "slot_box": (sx0, dock_y0, sx1, dock_y1),
                "piece_bbox": None,
            })
            uncertain_slots.append({
                "slot": s + 1,
                "name": slot_names[s],
                "reason": f"Không phát hiện thấy khối ở vị trí {slot_names[s]}",
            })

    board_mean_conf = float(np.mean(cell_confs))
    piece_mean_conf = float(np.mean([p["confidence"] for p in pieces])) if pieces else 0.0
    overall_conf = round(0.5 * board_mean_conf + 0.5 * piece_mean_conf, 3)

    meta = {
        "image_hash": img_hash,
        "bx": bx,
        "by": by,
        "bw": bw,
        "cw": cw,
        "dock_y0": dock_y0,
        "dock_y1": dock_y1,
        "dock_x0": dock_x0,
        "slot_w": slot_w,
        "scaled_img": scaled_img,
        "cell_confidences": cell_confs,
        "uncertain_cells": uncertain_cells,
        "uncertain_slots": uncertain_slots,
        "slot_details": slot_details,
        "overall_confidence": overall_conf,
    }
    return board, board_mask, pieces, meta


# ==============================================================================
# 2. CÔNG THỨC ĐIỂM SCORE THỰC TẾ & BỘ GIẢI VÉT CẠN TOÀN CỤC
# ==============================================================================
def precompute_placements(mat):
    if not mat or not mat[0]:
        return []
    R, C = len(mat), len(mat[0])
    cells = sum(sum(row) for row in mat)
    if cells == 0 or R > 8 or C > 8:
        return []
    placements = []
    for r0 in range(8 - R + 1):
        for c0 in range(8 - C + 1):
            pmask = 0
            for dr in range(R):
                for dc in range(C):
                    if mat[dr][dc] == 1:
                        pmask |= (1 << ((r0 + dr) * 8 + (c0 + dc)))
            placements.append((r0, c0, pmask, cells))
    return placements


def compute_verified_block_blast_score(cells_placed, lines_cleared, current_combo_streak):
    placement_pts = int(cells_placed)
    if lines_cleared <= 0:
        return placement_pts, 0, 0

    new_combo = current_combo_streak + 1
    base_clear_pts = 10 * lines_cleared * (lines_cleared + 1) // 2
    clear_pts = base_clear_pts * new_combo
    total_step_score = placement_pts + clear_pts
    return total_step_score, clear_pts, new_combo


def simulate_bitboard(bmask, pmask, cells, combo_streak=0):
    nb = bmask | pmask
    clear_mask = 0
    rows_cleared = 0
    cols_cleared = 0

    for rm in ROW_MASKS:
        if (nb & rm) == rm:
            clear_mask |= rm
            rows_cleared += 1
    for cm in COL_MASKS:
        if (nb & cm) == cm:
            clear_mask |= cm
            cols_cleared += 1

    lines = rows_cleared + cols_cleared
    if clear_mask:
        nb &= ~clear_mask

    step_score, _, next_combo = compute_verified_block_blast_score(
        cells, lines, combo_streak
    )
    if clear_mask and nb == 0:
        step_score += 50

    return nb, lines, rows_cleared, cols_cleared, step_score, next_combo


def evaluate_positional_heuristic(b):
    empty = 64 - b.bit_count()
    e = (~b) & FULL_MASK

    eh2 = e & (e >> 1) & NOT_RIGHT
    space_2x2 = (eh2 & (eh2 >> 8)).bit_count()

    eh3 = eh2 & (e >> 2) & 0x3F3F3F3F3F3F3F3F
    space_3x3 = (eh3 & (eh3 >> 8) & (eh3 >> 16)).bit_count()

    eh4 = eh2 & (eh2 >> 2) & 0x0F0F0F0F0F0F0F0F
    eh5 = eh4 & (e >> 4) & 0x0707070707070707
    ev2 = e & (e >> 8)
    ev4 = ev2 & (ev2 >> 16)
    ev5 = ev4 & (e >> 32)

    bars_4 = eh4.bit_count() + ev4.bit_count()
    bars_5 = eh5.bit_count() + ev5.bit_count()

    up_occ = ((b << 8) & FULL_MASK) | 0xFF
    down_occ = (b >> 8) | 0xFF00000000000000
    left_occ = ((b << 1) & NOT_LEFT) | 0x0101010101010101
    right_occ = ((b >> 1) & NOT_RIGHT) | 0x8080808080808080
    isolated_1x1 = (e & up_occ & down_occ & left_occ & right_occ).bit_count()

    near_clear_potential = 0.0
    for rm in ROW_MASKS:
        cnt = (b & rm).bit_count()
        if cnt == 7:
            near_clear_potential += 8.0
        elif cnt == 6:
            near_clear_potential += 3.5
    for cm in COL_MASKS:
        cnt = (b & cm).bit_count()
        if cnt == 7:
            near_clear_potential += 8.0
        elif cnt == 6:
            near_clear_potential += 3.5

    trans_h = ((b ^ (b >> 1)) & NOT_RIGHT).bit_count()
    trans_v = ((b ^ (b >> 8)) & 0x00FFFFFFFFFFFFFF).bit_count()

    return (
        empty * 12.0
        + space_3x3 * 24.0
        + space_2x2 * 6.5
        + bars_5 * 8.0
        + bars_4 * 4.5
        + near_clear_potential
        - isolated_1x1 * 45.0
        - (trans_h + trans_v) * 4.5
    )


def build_solution_ranking_key(steps, total_game_score, total_lines, max_simul_lines, heur_score):
    tie_breaker = tuple((-st["slot"], -st["row"], -st["col"]) for st in steps)
    return (
        len(steps),
        int(total_game_score),
        int(total_lines),
        int(max_simul_lines),
        round(float(heur_score), 4),
        tie_breaker,
    )


def solve_best_sequence(board_mask, pieces, return_details=False):
    t_start = time.perf_counter()
    if not pieces:
        if return_details:
            return {"best_steps": [], "backups": [], "stats": {"nodes": 0, "elapsed_ms": 0.0}}
        return []

    best_key = None
    best_solution = None
    top_candidates_map = {}
    nodes_evaluated = 0
    leaf_cache = {}

    def evaluate_last_piece_exact(p_last, cur_mask, combo_streak):
        nonlocal nodes_evaluated
        cache_key = (p_last["slot"], cur_mask, combo_streak)
        if cache_key in leaf_cache:
            return leaf_cache[cache_key]

        best_leaf = None
        best_leaf_metric = None

        for r0, c0, pmask, cells in p_last["placements"]:
            if (cur_mask & pmask) == 0:
                nodes_evaluated += 1
                nb, lines, r_clr, c_clr, step_score, next_combo = simulate_bitboard(
                    cur_mask, pmask, cells, combo_streak
                )
                h_val = evaluate_positional_heuristic(nb)
                metric = (
                    int(step_score),
                    int(lines),
                    int(lines),
                    round(float(h_val), 4),
                    -p_last["slot"],
                    -(r0 + 1),
                    -(c0 + 1),
                )
                if best_leaf_metric is None or metric > best_leaf_metric:
                    best_leaf_metric = metric
                    best_leaf = (r0, c0, nb, lines, r_clr, c_clr, step_score, next_combo, h_val)

        leaf_cache[cache_key] = best_leaf
        return best_leaf

    def register_solution(steps, final_mask, total_score, total_lines, max_simul):
        nonlocal best_key, best_solution
        if not steps:
            return
        h_val = evaluate_positional_heuristic(final_mask)
        rk = build_solution_ranking_key(steps, total_score, total_lines, max_simul, h_val)

        sol_obj = {
            "steps": [dict(s) for s in steps],
            "game_score": int(total_score),
            "total_lines": int(total_lines),
            "max_simultaneous_lines": int(max_simul),
            "heuristic_score": round(float(h_val), 2),
            "final_mask": final_mask,
            "rank_key": rk,
        }

        if best_key is None or rk > best_key:
            best_key = rk
            best_solution = sol_obj

        sig = tuple((s["slot"], s["row"], s["col"]) for s in steps)
        prev = top_candidates_map.get(sig)
        if prev is None or rk > prev["rank_key"]:
            top_candidates_map[sig] = sol_obj

    all_perms = list(itertools.permutations(pieces))
    visited_states = set()

    for perm in all_perms:
        n_p = len(perm)

        def dfs_exhaustive(idx, cur_mask, steps, combo_streak, acc_score, acc_lines, max_simul):
            nonlocal nodes_evaluated
            if idx == n_p:
                register_solution(steps, cur_mask, acc_score, acc_lines, max_simul)
                return

            rem_slots = tuple(perm[k]["slot"] for k in range(idx, n_p))
            state_sig = (rem_slots, cur_mask, combo_streak, acc_score, acc_lines, max_simul, len(steps))
            if state_sig in visited_states:
                return
            visited_states.add(state_sig)

            p = perm[idx]

            if idx == n_p - 1:
                leaf_res = evaluate_last_piece_exact(p, cur_mask, combo_streak)
                if leaf_res is not None:
                    r0, c0, nb, lines, r_clr, c_clr, step_score, _, _ = leaf_res
                    steps.append({
                        "slot": p["slot"],
                        "name": p["name"],
                        "matrix": p["matrix"],
                        "row": r0 + 1,
                        "col": c0 + 1,
                        "lines": lines,
                        "rows_cleared": r_clr,
                        "cols_cleared": c_clr,
                        "step_score": step_score,
                    })
                    register_solution(
                        steps,
                        nb,
                        acc_score + step_score,
                        acc_lines + lines,
                        max(max_simul, lines),
                    )
                    steps.pop()
                else:
                    register_solution(steps, cur_mask, acc_score, acc_lines, max_simul)
                return

            any_placed = False
            for r0, c0, pmask, cells in p["placements"]:
                if (cur_mask & pmask) == 0:
                    any_placed = True
                    nodes_evaluated += 1
                    nb, lines, r_clr, c_clr, step_score, next_combo = simulate_bitboard(
                        cur_mask, pmask, cells, combo_streak
                    )
                    steps.append({
                        "slot": p["slot"],
                        "name": p["name"],
                        "matrix": p["matrix"],
                        "row": r0 + 1,
                        "col": c0 + 1,
                        "lines": lines,
                        "rows_cleared": r_clr,
                        "cols_cleared": c_clr,
                        "step_score": step_score,
                    })
                    dfs_exhaustive(
                        idx + 1,
                        nb,
                        steps,
                        next_combo,
                        acc_score + step_score,
                        acc_lines + lines,
                        max(max_simul, lines),
                    )
                    steps.pop()

            if not any_placed:
                register_solution(steps, cur_mask, acc_score, acc_lines, max_simul)
                dfs_exhaustive(idx + 1, cur_mask, steps, combo_streak, acc_score, acc_lines, max_simul)

        dfs_exhaustive(0, board_mask, [], 0, 0, 0, 0)

    elapsed_ms = round((time.perf_counter() - t_start) * 1000.0, 2)
    sorted_all = sorted(top_candidates_map.values(), key=lambda x: x["rank_key"], reverse=True)
    best_steps = best_solution["steps"] if best_solution else []

    if return_details:
        explanation = build_solution_explanation(best_solution, len(pieces), nodes_evaluated, elapsed_ms)
        return {
            "best_steps": best_steps,
            "best_solution": best_solution,
            "backups": sorted_all[1:4],
            "explanation": explanation,
            "stats": {
                "nodes_evaluated": nodes_evaluated,
                "elapsed_ms": elapsed_ms,
                "unique_sequences_found": len(sorted_all),
            },
        }

    return best_steps


def build_solution_explanation(best_sol, total_pieces, nodes_evaluated, elapsed_ms):
    if not best_sol or not best_sol["steps"]:
        return (
            f"Đã kiểm tra vét cạn 100% ({nodes_evaluated} trạng thái trong {elapsed_ms}ms) "
            "nhưng không tồn tại vị trí hợp lệ nào cho cả 3 khối."
        )

    steps = best_sol["steps"]
    placed_cnt = len(steps)
    g_score = best_sol["game_score"]
    t_lines = best_sol["total_lines"]
    h_score = best_sol["heuristic_score"]
    order_str = " ➔ ".join(f"{s['name']} (H{s['row']},C{s['col']})" for s in steps)

    return (
        f"Đặt được {placed_cnt}/{total_pieces} khối theo thứ tự: {order_str}. "
        f"Đạt tổng điểm GAME SCORE cao nhất là +{g_score} điểm (xóa {t_lines} hàng/cột). "
        f"Điểm thế cờ: {h_score}. Đã duyệt vét cạn {nodes_evaluated} trạng thái trong {elapsed_ms}ms."
    )


# ==============================================================================
# 3. ĐỊNH DẠNG BÁO CÁO, THÔNG BÁO iOS & VẼ ẢNH KẾT QUẢ
# ==============================================================================
def format_matrix_str(mat):
    if not mat:
        return "(Trống)"
    return "\n".join(" ".join(str(cell) for cell in row) for row in mat)


def format_verification_report_text(board, pieces, meta):
    lines = []
    lines.append("🔍 KẾT QUẢ NHẬN DIỆN DỮ LIỆU ĐẦU VÀO")
    lines.append(f"• Mã ảnh (SHA256): {meta['image_hash']}")
    lines.append(f"• Độ tin cậy tổng thể: {meta['overall_confidence'] * 100:.1f}%")

    if meta["uncertain_slots"] or meta["uncertain_cells"]:
        lines.append("⚠️ CẢNH BÁO VÙNG CHƯA RÕ RÀNG:")
        for us in meta["uncertain_slots"]:
            lines.append(f"  - Khối {us['slot']} ({us['name']}): {us['reason']}")
        if meta["uncertain_cells"]:
            coords = ", ".join(f"(H{c['row']},C{c['col']})" for c in meta["uncertain_cells"][:6])
            lines.append(f"  - Ô bàn cờ cần kiểm tra lại: {coords}")
    else:
        lines.append("✅ Trạng thái: Nhận diện rõ ràng, độ tin cậy cao.")

    lines.append("\nBàn cờ (8x8):")
    lines.append("   C1 C2 C3 C4 C5 C6 C7 C8")
    for r in range(8):
        row_str = "  ".join(str(board[r][c]) for c in range(8))
        lines.append(f"H{r+1}  {row_str}")

    lines.append("\nBa khối đang chờ:")
    slot_map = {p["slot"]: p for p in pieces}
    slot_names = ["TRÁI", "GIỮA", "PHẢI"]
    for s in range(1, 4):
        p = slot_map.get(s)
        lines.append(f"Khối {s} ({slot_names[s-1]}):")
        if p:
            R, C = len(p["matrix"]), len(p["matrix"][0])
            lines.append(f"[Kích thước {R}x{C} | Độ tin cậy: {p['confidence']*100:.1f}%]")
            lines.append(format_matrix_str(p["matrix"]))
        else:
            lines.append("(Không nhận diện được / Ô trống)")
        lines.append("")

    return "\n".join(lines).strip()


def format_ios_alert_text(board, steps, details=None, meta=None):
    warning_header = ""
    if meta and (meta.get("uncertain_slots") or len(meta.get("uncertain_cells", [])) > 3):
        warning_header = "⚠️ LƯU Ý: Có vùng ảnh độ tương phản thấp, hãy kiểm tra kỹ!\n"

    if not steps:
        return warning_header + "⚠️ Không tìm thấy nước đi hợp lệ!"

    grid = [["🟦" if board[r][c] == 1 else "⬛" for c in range(8)] for r in range(8)]
    step_badges = ["1️⃣", "2️⃣", "3️⃣"]
    lines_text = []

    for i, st in enumerate(steps):
        badge = step_badges[i] if i < 3 else "🔸"
        clear_note = f" (💥Xóa {st['lines']} | +{st.get('step_score', 0)}đ)" if st["lines"] > 0 else f" (+{st.get('step_score', 0)}đ)"
        lines_text.append(
            f"{badge} {st['name']} ➔ Hàng {st['row']}, Cột {st['col']}{clear_note}"
        )

        mat = st["matrix"]
        r0, c0 = st["row"] - 1, st["col"] - 1
        for dr in range(len(mat)):
            for dc in range(len(mat[0])):
                if mat[dr][dc] == 1 and 0 <= r0 + dr < 8 and 0 <= c0 + dc < 8:
                    grid[r0 + dr][c0 + dc] = badge

    score_line = ""
    if details and details.get("best_solution"):
        bs = details["best_solution"]
        score_line = (
            f"🏆 Tổng SCORE: +{bs['game_score']} điểm | "
            f"Xóa: {bs['total_lines']} hàng/cột | "
            f"Thế cờ: {bs['heuristic_score']}"
        )

    map_str = "\n".join("".join(row) for row in grid)
    out_parts = []
    if warning_header:
        out_parts.append(warning_header.strip())
    out_parts.extend(lines_text)
    if score_line:
        out_parts.append(score_line)
    out_parts.append(map_str)
    return "\n".join(out_parts)


def render_inspection_image(board, pieces, meta):
    img = meta["scaled_img"].copy()
    draw = ImageDraw.Draw(img, "RGBA")
    bx, by, bw, cw = meta["bx"], meta["by"], meta["bw"], meta["cw"]

    draw.rectangle([bx - 2, by - 2, bx + bw + 2, by + bw + 2], outline=(0, 255, 180, 255), width=3)
    for c in range(8):
        cx = int(round(bx + (c + 0.35) * cw))
        draw.text((cx, max(4, by - 22)), f"C{c+1}", fill=(255, 255, 255, 255))
    for r in range(8):
        ry = int(round(by + (r + 0.35) * cw))
        draw.text((max(2, bx - 26), ry), f"H{r+1}", fill=(255, 255, 255, 255))

    for r in range(8):
        for c in range(8):
            x0 = int(round(bx + c * cw))
            y0 = int(round(by + r * cw))
            x1 = int(round(bx + (c + 1) * cw))
            y1 = int(round(by + (r + 1) * cw))
            draw.rectangle([x0, y0, x1, y1], outline=(255, 255, 255, 70), width=1)
            if board[r][c] == 1:
                draw.rectangle(
                    [x0 + 8, y0 + 8, x1 - 8, y1 - 8],
                    fill=(34, 197, 94, 110),
                    outline=(74, 222, 128, 240),
                    width=2,
                )
                draw.text((x0 + 12, y0 + 10), "1", fill=(255, 255, 255, 255))
            else:
                draw.text((x0 + 12, y0 + 10), "0", fill=(160, 174, 192, 180))

    for sd in meta.get("slot_details", []):
        sx0, sy0, sx1, sy1 = sd["slot_box"]
        draw.rectangle([sx0, sy0, sx1, sy1], outline=(148, 163, 184, 140), width=2)
        if sd["detected"] and sd["piece_bbox"]:
            min_x, min_y, max_x, max_y = sd["piece_bbox"]
            mat = sd["matrix"]
            rows, cols = len(mat), len(mat[0])
            px0, py0 = sx0 + min_x, sy0 + min_y
            px1, py1 = sx0 + max_x, sy0 + max_y
            sub_w = (px1 - px0 + 1) / float(cols)
            sub_h = (py1 - py0 + 1) / float(rows)

            for r in range(rows):
                for c in range(cols):
                    cx0 = int(round(px0 + c * sub_w))
                    cy0 = int(round(py0 + r * sub_h))
                    cx1 = int(round(px0 + (c + 1) * sub_w))
                    cy1 = int(round(py0 + (r + 1) * sub_h))
                    if mat[r][c] == 1:
                        draw.rectangle(
                            [cx0, cy0, cx1, cy1],
                            fill=(56, 189, 248, 100),
                            outline=(0, 255, 255, 255),
                            width=2,
                        )
                    else:
                        draw.rectangle([cx0, cy0, cx1, cy1], outline=(255, 100, 100, 130), width=1)

            label = f"{sd['name']}: {rows}x{cols} ({sd['confidence']*100:.0f}%)"
            draw.rectangle([sx0 + 4, sy0 + 4, sx1 - 4, sy0 + 24], fill=(15, 23, 42, 210))
            draw.text((sx0 + 8, sy0 + 7), label, fill=(56, 189, 248, 255))

    buf = io.BytesIO()
    img.save(buf, format="PNG")
    buf.seek(0)
    return buf


def render_solution_image(meta, steps, details=None, board=None):
    img = meta["scaled_img"].copy()
    draw = ImageDraw.Draw(img, "RGBA")
    bx, by, bw, cw = meta["bx"], meta["by"], meta["bw"], meta["cw"]

    for c in range(8):
        cx = int(round(bx + (c + 0.38) * cw))
        draw.text((cx, max(4, by - 20)), f"C{c+1}", fill=(255, 255, 255, 255))
    for r in range(8):
        ry = int(round(by + (r + 0.38) * cw))
        draw.text((max(2, bx - 24), ry), f"H{r+1}", fill=(255, 255, 255, 255))

    colors = [
        (56, 189, 248, 195),
        (250, 204, 21, 195),
        (192, 132, 252, 195),
    ]

    for i, st in enumerate(steps):
        color = colors[i % len(colors)]
        mat = st["matrix"]
        r0, c0 = st["row"] - 1, st["col"] - 1
        for dr in range(len(mat)):
            for dc in range(len(mat[0])):
                if mat[dr][dc] == 1:
                    x0 = int(round(bx + (c0 + dc) * cw + 3))
                    y0 = int(round(by + (r0 + dr) * cw + 3))
                    x1 = int(round(bx + (c0 + dc + 1) * cw - 3))
                    y1 = int(round(by + (r0 + dr + 1) * cw - 3))
                    draw.rectangle([x0, y0, x1, y1], fill=color, outline=(255, 255, 255, 255), width=3)
                    draw.text((x0 + 6, y0 + 5), f"#{i+1}", fill=(10, 15, 30, 255))

    banner_y0 = max(6, by - 135)
    banner_y1 = max(banner_y0 + 88, by - 26)
    draw.rectangle([bx, banner_y0, bx + bw, banner_y1], fill=(15, 23, 42, 230), outline=(56, 189, 248, 255), width=2)

    if not steps:
        draw.text((bx + 14, banner_y0 + 28), "KHÔNG TÌM THẤY NƯỚC ĐI HỢP LỆ!", fill=(248, 113, 113, 255))
    else:
        total_sc = details["best_solution"]["game_score"] if (details and details.get("best_solution")) else 0
        draw.text(
            (bx + 12, banner_y0 + 8),
            f"PHƯƠNG ÁN TỐI ƯU SCORE: +{total_sc} ĐIỂM (Độ tin cậy: {meta['overall_confidence']*100:.1f}%)",
            fill=(74, 222, 128, 255),
        )
        for idx, st in enumerate(steps):
            clr_txt = f" [Xóa {st['lines']} hàng/cột]" if st["lines"] > 0 else ""
            line_str = f"Bước #{idx+1}: Khối {st['name']} -> Đặt tại Hàng {st['row']}, Cột {st['col']}{clr_txt}"
            draw.text((bx + 12, banner_y0 + 30 + idx * 20), line_str, fill=(255, 255, 255, 255))

    buf = io.BytesIO()
    img.save(buf, format="PNG")
    buf.seek(0)
    return buf


# ==============================================================================
# 4. LƯU LOG LỊCH SỬ & TRÍCH XUẤT ẢNH TỪ REQUEST
# ==============================================================================
def save_audit_log(meta, board, pieces, details):
    try:
        record = {
            "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
            "image_hash": meta.get("image_hash", ""),
            "overall_confidence": meta.get("overall_confidence", 0.0),
            "board": board,
            "pieces": [{"slot": p["slot"], "name": p["name"], "matrix": p["matrix"]} for p in pieces],
            "best_steps": details.get("best_steps", []),
            "game_score": details["best_solution"]["game_score"] if details.get("best_solution") else 0,
            "stats": details.get("stats", {}),
        }
        with open(HISTORY_LOG_PATH, "a", encoding="utf-8") as f:
            f.write(json.dumps(record, ensure_ascii=False) + "\n")
    except Exception:
        pass


def extract_image_from_request(req):
    if req.files:
        f = next(iter(req.files.values()))
        return Image.open(f.stream).convert("RGB")
    if req.data and len(req.data) > 100 and not req.is_json:
        return Image.open(io.BytesIO(req.data)).convert("RGB")
    if req.is_json and req.json and "image" in req.json:
        b64 = req.json["image"].split(",")[-1]
        return Image.open(io.BytesIO(base64.b64decode(b64))).convert("RGB")
    return None


# ==============================================================================
# 5. CÁC ENDPOINT FLASK SERVER (/img, /solve, /inspect, /solve_manual, /ui)
# ==============================================================================
@app.route("/", methods=["GET"])
def health_check():
    if request.args.get("ui") == "1":
        return render_web_inspector_ui()
    return "OK", 200


@app.route("/img", methods=["POST"])
def solve_image_shortcut_endpoint():
    """Đường dẫn tắt siêu gọn cho iOS Shortcut: Gửi ảnh lên là trả về ngay ảnh PNG giải sẵn."""
    try:
        pil_img = extract_image_from_request(request)
        if pil_img is None:
            return "⚠️ Không nhận được ảnh!", 400
        board, board_mask, pieces, meta = detect_board_and_pieces(pil_img)
        details = solve_best_sequence(board_mask, pieces, return_details=True)
        steps = details["best_steps"]
        save_audit_log(meta, board, pieces, details)
        buf = render_solution_image(meta, steps, details=details, board=board)
        return send_file(buf, mimetype="image/png")
    except Exception as e:
        return f"❌ Lỗi: {str(e)}", 500


@app.route("/solve", methods=["POST"])
def solve_endpoint():
    try:
        pil_img = extract_image_from_request(request)
        if pil_img is None:
            return "⚠️ Không nhận được ảnh!", 400

        board, board_mask, pieces, meta = detect_board_and_pieces(pil_img)
        fmt = request.args.get("format", "text").lower()

        session_id = str(uuid.uuid4())[:8]
        SESSIONS_CACHE["latest"] = {
            "session_id": session_id,
            "board": board,
            "pieces": pieces,
            "meta": meta,
        }
        SESSIONS_CACHE[session_id] = SESSIONS_CACHE["latest"]

        if fmt == "verify_text":
            report = format_verification_report_text(board, pieces, meta)
            return report, 200, {"Content-Type": "text/plain; charset=utf-8"}

        if fmt == "inspect_image":
            buf = render_inspection_image(board, pieces, meta)
            return send_file(buf, mimetype="image/png")

        details = solve_best_sequence(board_mask, pieces, return_details=True)
        steps = details["best_steps"]
        save_audit_log(meta, board, pieces, details)

        if fmt == "image":
            buf = render_solution_image(meta, steps, details=details, board=board)
            return send_file(buf, mimetype="image/png")

        if fmt == "json":
            sol_buf = render_solution_image(meta, steps, details=details, board=board)
            insp_buf = render_inspection_image(board, pieces, meta)
            return jsonify({
                "session_id": session_id,
                "image_hash": meta["image_hash"],
                "overall_confidence": meta["overall_confidence"],
                "uncertain_cells": meta["uncertain_cells"],
                "uncertain_slots": meta["uncertain_slots"],
                "board": board,
                "pieces": [
                    {"slot": p["slot"], "name": p["name"], "matrix": p["matrix"], "confidence": p["confidence"]}
                    for p in pieces
                ],
                "verification_text": format_verification_report_text(board, pieces, meta),
                "alert_text": format_ios_alert_text(board, steps, details=details, meta=meta),
                "best_steps": steps,
                "game_score": details["best_solution"]["game_score"] if details["best_solution"] else 0,
                "heuristic_score": details["best_solution"]["heuristic_score"] if details["best_solution"] else 0.0,
                "explanation": details["explanation"],
                "backups": [
                    {"steps": b["steps"], "game_score": b["game_score"], "total_lines": b["total_lines"]}
                    for b in details["backups"]
                ],
                "stats": details["stats"],
                "inspection_image_b64": base64.b64encode(insp_buf.getvalue()).decode("ascii"),
                "solution_image_b64": base64.b64encode(sol_buf.getvalue()).decode("ascii"),
            })

        alert_text = format_ios_alert_text(board, steps, details=details, meta=meta)
        return alert_text, 200, {"Content-Type": "text/plain; charset=utf-8"}

    except Exception as e:
        return f"❌ Lỗi: {str(e)}", 500


@app.route("/inspect", methods=["POST"])
def inspect_endpoint():
    try:
        pil_img = extract_image_from_request(request)
        if pil_img is None:
            return "⚠️ Không nhận được ảnh!", 400
        board, _, pieces, meta = detect_board_and_pieces(pil_img)
        session_id = str(uuid.uuid4())[:8]
        SESSIONS_CACHE["latest"] = {"session_id": session_id, "board": board, "pieces": pieces, "meta": meta}
        SESSIONS_CACHE[session_id] = SESSIONS_CACHE["latest"]

        if request.args.get("format") == "image":
            buf = render_inspection_image(board, pieces, meta)
            return send_file(buf, mimetype="image/png")

        report = format_verification_report_text(board, pieces, meta)
        return report, 200, {"Content-Type": "text/plain; charset=utf-8"}
    except Exception as e:
        return f"❌ Lỗi: {str(e)}", 500


@app.route("/solve_manual", methods=["POST"])
def solve_manual_endpoint():
    try:
        data = request.get_json(force=True)
        board = data.get("board")
        raw_pieces = data.get("pieces", [])

        if not board or len(board) != 8 or any(len(row) != 8 for row in board):
            return jsonify({"error": "Ma trận bàn cờ phải có kích thước đúng 8x8!"}), 400

        board_mask = 0
        for r in range(8):
            for c in range(8):
                if int(board[r][c]) == 1:
                    board_mask |= (1 << (r * 8 + c))

        slot_names = ["TRÁI", "GIỮA", "PHẢI"]
        pieces = []
        for idx, rp in enumerate(raw_pieces):
            mat = rp.get("matrix", [])
            if not mat or not any(any(int(v) == 1 for v in row) for row in mat):
                continue
            clean_mat = [[1 if int(v) == 1 else 0 for v in row] for row in mat]
            arr_m = np.array(clean_mat, dtype=int)
            vr = np.any(arr_m == 1, axis=1)
            vc = np.any(arr_m == 1, axis=0)
            trimmed = arr_m[vr][:, vc].tolist()
            s_num = int(rp.get("slot", idx + 1))
            s_name = rp.get("name", slot_names[(s_num - 1) % 3])
            pieces.append({
                "slot": s_num,
                "name": s_name,
                "matrix": trimmed,
                "confidence": 1.0,
                "placements": precompute_placements(trimmed),
            })

        details = solve_best_sequence(board_mask, pieces, return_details=True)
        steps = details["best_steps"]
        alert_text = format_ios_alert_text(board, steps, details=details, meta=None)

        sol_img_b64 = None
        if "latest" in SESSIONS_CACHE and SESSIONS_CACHE["latest"].get("meta"):
            buf = render_solution_image(
                SESSIONS_CACHE["latest"]["meta"], steps, details=details, board=board
            )
            sol_img_b64 = base64.b64encode(buf.getvalue()).decode("ascii")

        return jsonify({
            "alert_text": alert_text,
            "best_steps": steps,
            "game_score": details["best_solution"]["game_score"] if details["best_solution"] else 0,
            "heuristic_score": details["best_solution"]["heuristic_score"] if details["best_solution"] else 0.0,
            "explanation": details["explanation"],
            "backups": [
                {"steps": b["steps"], "game_score": b["game_score"], "total_lines": b["total_lines"]}
                for b in details["backups"]
            ],
            "stats": details["stats"],
            "solution_image_b64": sol_img_b64,
        })
    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route("/ui", methods=["GET"])
def render_web_inspector_ui():
    html = """<!DOCTYPE html>
<html lang="vi">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>Block Blast Solver - Kiểm Chứng & Giải Tối Ưu SCORE</title>
<style>
  body { font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, sans-serif; background: #0f172a; color: #f8fafc; margin: 0; padding: 16px; }
  h1, h2, h3 { margin-top: 0; color: #38bdf8; }
  .container { max-width: 1080px; margin: 0 auto; display: grid; grid-template-columns: repeat(auto-fit, minmax(330px, 1fr)); gap: 18px; }
  .card { background: #1e293b; border: 1px solid #334155; border-radius: 12px; padding: 16px; }
  .btn { background: #0284c7; color: #fff; border: none; padding: 10px 16px; border-radius: 8px; font-weight: 600; cursor: pointer; margin-right: 8px; margin-top: 8px; }
  .btn-success { background: #16a34a; }
  .btn:hover { opacity: 0.92; }
  .board-grid { display: grid; grid-template-columns: repeat(8, 34px); gap: 4px; margin: 10px 0; }
  .cell { width: 34px; height: 34px; background: #0f172a; border: 1px solid #475569; border-radius: 5px; display: flex; align-items: center; justify-content: center; cursor: pointer; font-weight: bold; user-select: none; }
  .cell.on { background: #3b82f6; border-color: #93c5fd; color: #fff; }
  .piece-grid { display: grid; grid-template-columns: repeat(5, 28px); gap: 3px; margin: 6px 0 14px 0; }
  .pcell { width: 28px; height: 28px; background: #0f172a; border: 1px solid #475569; border-radius: 4px; cursor: pointer; }
  .pcell.on { background: #f59e0b; border-color: #fde68a; }
  pre { background: #090d16; padding: 12px; border-radius: 8px; overflow-x: auto; font-size: 13px; line-height: 1.45; border: 1px solid #334155; }
  img.preview { max-width: 100%; border-radius: 8px; border: 1px solid #475569; margin-top: 8px; }
  .badge { display: inline-block; padding: 4px 8px; border-radius: 6px; background: #334155; font-size: 12px; margin-bottom: 8px; }
</style>
</head>
<body>
  <div style="max-width:1080px;margin:0 auto 16px auto;">
    <h1>🧩 Block Blast Solver – Kiểm Chứng Nhận Diện & Tối Ưu SCORE</h1>
    <p style="color:#94a3b8;margin:4px 0;">Tải ảnh chụp màn hình lên để kiểm tra bàn cờ 8×8 và 3 khối hình. Nếu muốn chỉnh sửa, hãy chạm trực tiếp vào các ô bên dưới rồi bấm <b>Giải Lại Từ Ma Trận Đang Chỉnh Sửa</b>.</p>
    <input type="file" id="fileInput" accept="image/*" style="margin-top:8px;">
    <button class="btn" onclick="uploadAndInspect()">1. Quét Ảnh & Giải Ngay</button>
    <button class="btn btn-success" onclick="solveFromManualGrid()">2. Giải Lại Từ Ma Trận Đang Chỉnh Sửa</button>
  </div>

  <div class="container">
    <div class="card">
      <h3>A. Chỉnh Sửa Thủ Công Bàn Cờ 8×8</h3>
      <span class="badge" id="confBadge">Độ tin cậy: --</span>
      <div id="boardGrid" class="board-grid"></div>
      <h3>B. Chỉnh Sửa 3 Khối Đang Chờ (Lưới 5×5)</h3>
      <div><b>Khối 1 (TRÁI):</b><div id="piece0" class="piece-grid"></div></div>
      <div><b>Khối 2 (GIỮA):</b><div id="piece1" class="piece-grid"></div></div>
      <div><b>Khối 3 (PHẢI):</b><div id="piece2" class="piece-grid"></div></div>
    </div>

    <div class="card">
      <h3>C. Kết Quả Giải & Điểm SCORE Dự Kiến</h3>
      <pre id="solutionText">Chưa có dữ liệu. Hãy chọn ảnh chụp màn hình và bấm "Quét Ảnh & Giải Ngay".</pre>
      <h3>D. Báo Cáo Kiểm Chứng Ma Trận (0/1)</h3>
      <pre id="verifyText">--</pre>
    </div>

    <div class="card">
      <h3>E. Ảnh Đánh Dấu Vị Trí Đặt & Lưới Nhận Diện</h3>
      <div id="imgContainer"></div>
    </div>
  </div>

<script>
let boardState = Array.from({length: 8}, () => Array(8).fill(0));
let piecesState = [
  Array.from({length: 5}, () => Array(5).fill(0)),
  Array.from({length: 5}, () => Array(5).fill(0)),
  Array.from({length: 5}, () => Array(5).fill(0))
];

function renderGrids() {
  const bDiv = document.getElementById('boardGrid');
  bDiv.innerHTML = '';
  for (let r = 0; r < 8; r++) {
    for (let c = 0; c < 8; c++) {
      const el = document.createElement('div');
      el.className = 'cell' + (boardState[r][c] ? ' on' : '');
      el.textContent = boardState[r][c];
      el.onclick = () => { boardState[r][c] ^= 1; renderGrids(); };
      bDiv.appendChild(el);
    }
  }
  for (let s = 0; s < 3; s++) {
    const pDiv = document.getElementById('piece' + s);
    pDiv.innerHTML = '';
    for (let r = 0; r < 5; r++) {
      for (let c = 0; c < 5; c++) {
        const el = document.createElement('div');
        el.className = 'pcell' + (piecesState[s][r][c] ? ' on' : '');
        el.onclick = () => { piecesState[s][r][c] ^= 1; renderGrids(); };
        pDiv.appendChild(el);
      }
    }
  }
}

async function uploadAndInspect() {
  const fi = document.getElementById('fileInput');
  if (!fi.files.length) { alert('Vui lòng chọn ảnh chụp màn hình Block Blast!'); return; }
  const fd = new FormData();
  fd.append('image', fi.files[0]);
  document.getElementById('solutionText').textContent = 'Đang phân tích ảnh và tìm phương án SCORE cao nhất...';
  const res = await fetch('/solve?format=json', { method: 'POST', body: fd });
  const data = await res.json();
  boardState = data.board;
  piecesState = [
    Array.from({length: 5}, () => Array(5).fill(0)),
    Array.from({length: 5}, () => Array(5).fill(0)),
    Array.from({length: 5}, () => Array(5).fill(0))
  ];
  for (const p of data.pieces) {
    const idx = p.slot - 1;
    const mat = p.matrix;
    for (let r = 0; r < mat.length; r++) {
      for (let c = 0; c < mat[0].length; c++) {
        piecesState[idx][r][c] = mat[r][c];
      }
    }
  }
  renderGrids();
  document.getElementById('confBadge').textContent = 'Độ tin cậy: ' + (data.overall_confidence * 100).toFixed(1) + '% | Hash: ' + data.image_hash;
  document.getElementById('solutionText').textContent = data.alert_text + '\\n\\nGiải thích: ' + data.explanation;
  document.getElementById('verifyText').textContent = data.verification_text;
  document.getElementById('imgContainer').innerHTML =
    '<b>Ảnh hướng dẫn đặt khối:</b><br><img class="preview" src="data:image/png;base64,' + data.solution_image_b64 + '"><br><br>' +
    '<b>Ảnh kiểm chứng lưới nhận diện:</b><br><img class="preview" src="data:image/png;base64,' + data.inspection_image_b64 + '">';
}

async function solveFromManualGrid() {
  const names = ['TRÁI', 'GIỮA', 'PHẢI'];
  const payload = {
    board: boardState,
    pieces: [0, 1, 2].map(i => ({ slot: i + 1, name: names[i], matrix: piecesState[i] }))
  };
  document.getElementById('solutionText').textContent = 'Đang tính toán vét cạn từ ma trận đã chỉnh sửa...';
  const res = await fetch('/solve_manual', {
    method: 'POST',
    headers: {'Content-Type': 'application/json'},
    body: JSON.stringify(payload)
  });
  const data = await res.json();
  document.getElementById('solutionText').textContent = data.alert_text + '\\n\\nGiải thích: ' + data.explanation;
  if (data.solution_image_b64) {
    document.getElementById('imgContainer').innerHTML =
      '<b>Ảnh hướng dẫn đặt khối (Đã cập nhật theo chỉnh sửa):</b><br><img class="preview" src="data:image/png;base64,' + data.solution_image_b64 + '">';
  }
}

renderGrids();
</script>
</body>
</html>"""
    return Response(html, mimetype="text/html; charset=utf-8")


if __name__ == "__main__":
    port = int(os.environ.get("PORT", 10000))
    app.run(host="0.0.0.0", port=port)