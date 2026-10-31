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
# CẤU HÌNH LƯU TRỮ PHIÊN KIỂM CHỨNG & LỊCH SỬ GIẢI (ĐẢM BẢO TÁI LẬP KẾT QUẢ)
# ==============================================================================
DATA_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "solver_data")
os.makedirs(DATA_DIR, exist_ok=True)
HISTORY_LOG_PATH = os.path.join(DATA_DIR, "solver_history.jsonl")
SESSIONS_CACHE = {}

# ==============================================================================
# HẰNG SỐ BITBOARD 64-BIT
# Ô (r, c) với r in [0..7], c in [0..7] ứng với bit: (r * 8 + c)
# ==============================================================================
FULL_MASK = 0xFFFFFFFFFFFFFFFF
NOT_RIGHT = 0x7F7F7F7F7F7F7F7F
NOT_LEFT = 0xFEFEFEFEFEFEFEFE
ROW_MASKS = [0xFF << (r * 8) for r in range(8)]
COL_MASKS = [0x0101010101010101 << c for c in range(8)]


# ==============================================================================
# 1. HỆ THỐNG XỬ LÝ ẢNH & NHẬN DIỆN ĐỘ CHÍNH XÁC CAO (HIGH-RES GEOMETRIC VISION)
# ==============================================================================
def compute_image_sha256(pil_img):
    """Tạo mã băm cố định cho ảnh đầu vào để kiểm chứng tính bất biến."""
    raw_bytes = pil_img.convert("RGB").tobytes()
    return hashlib.sha256(raw_bytes).hexdigest()[:16]


def normalize_image(pil_img, target_w=720):
    """
    Chuẩn hóa ảnh về chiều rộng 720px (gấp đôi bản cũ) để giữ nguyên độ nét
    của các rãnh ngăn cách 2-4px giữa các ô gạch trong khu vực chờ.
    """
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
    """Tính kênh độ sáng (Luminance) và độ bão hòa màu (Chroma) từ mảng RGB."""
    lum = 0.299 * arr[:, :, 0] + 0.587 * arr[:, :, 1] + 0.114 * arr[:, :, 2]
    chroma = arr.max(axis=2) - arr.min(axis=2)
    return lum, chroma


def auto_locate_board(rgb_arr):
    """
    Định vị chính xác khung bàn cờ 8x8 dựa trên:
    1. Đạo hàm biên dọc & ngang của khung bàn cờ.
    2. Tính đối xứng trái-phải của màn hình.
    3. Điểm cộng hưởng chu kỳ của 7 đường lưới dọc và 7 đường lưới ngang nội bộ.
    """
    h, w, _ = rgb_arr.shape
    lum, _ = rgb_to_lum_chroma(rgb_arr)

    vert_grad = np.abs(np.diff(lum, axis=0, prepend=lum[:1, :]))
    horiz_grad = np.abs(np.diff(lum, axis=1, prepend=lum[:, :1]))

    # 1. Tìm biên trái (bx) và chiều rộng bàn cờ (bw)
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

        # Cộng điểm cho 7 đường chia cột bên trong bàn cờ
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

    # 2. Tìm tọa độ đỉnh trên của bàn cờ (by)
    row_edge_profile = vert_grad[:, best_bx : best_bx + best_bw].mean(axis=1)
    row_lum_profile = lum[:, best_bx : best_bx + best_bw].mean(axis=1)

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

        # Vùng bên trong bàn cờ thường có viền tối rõ rệt ở mép trên/dưới
        score_y = top_peak + bot_peak * 0.92 + (internal_h / 7.0) * 0.85
        if score_y > best_y_score:
            best_y_score = score_y
            best_by = y

    return best_bx, best_by, best_bw


def detect_board_cells(rgb_arr, bx, by, bw):
    """
    Phân tích 64 ô trên bàn cờ 8x8.
    Trả về:
      - board: ma trận 8x8 (0 hoặc 1)
      - board_mask: số nguyên 64-bit
      - cell_confidences: ma trận 8x8 chứa độ tin cậy [0.0 .. 1.0] của từng ô
      - uncertain_cells: danh sách các ô có độ tin cậy thấp (< 0.72)
    """
    cw = bw / 8.0
    lum_full, chroma_full = rgb_to_lum_chroma(rgb_arr)

    # Lấy mẫu màu rãnh lưới (mortar lines) giữa các ô để làm mốc nền tối chuẩn
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

            # Vùng lõi ô (bỏ 14% rìa ngoài để tránh đường kẻ lưới)
            pad_x = max(2, int(round(cell_w * 0.14)))
            pad_y = max(2, int(round(cell_h * 0.14)))
            core_rgb = rgb_arr[y0 + pad_y : y1 - pad_y, x0 + pad_x : x1 - pad_x]
            core_lum = lum_full[y0 + pad_y : y1 - pad_y, x0 + pad_x : x1 - pad_x]
            core_chroma = chroma_full[y0 + pad_y : y1 - pad_y, x0 + pad_x : x1 - pad_x]

            # Vùng tâm phẳng (30%..70%) để đo màu mặt gạch chính xác
            in_x = max(3, int(round(cell_w * 0.28)))
            in_y = max(3, int(round(cell_h * 0.28)))
            center_rgb = rgb_arr[y0 + in_y : y1 - in_y, x0 + in_x : x1 - in_x]
            mean_rgb = center_rgb.mean(axis=(0, 1)) if center_rgb.size > 0 else core_rgb.mean(axis=(0, 1))

            std_lum = float(core_lum.std()) if core_lum.size > 0 else 0.0
            mean_lum = float(core_lum.mean()) if core_lum.size > 0 else 0.0
            mean_chroma = float(core_chroma.mean()) if core_chroma.size > 0 else 0.0

            # Đo hiệu ứng vát cạnh 3D (Bevel): gạch thật luôn có gờ trên/trái sáng và gờ dưới/phải tối
            b_strip = max(2, int(round(core_lum.shape[0] * 0.22)))
            top_l = float(core_lum[:b_strip, :].mean())
            bot_l = float(core_lum[-b_strip:, :].mean())
            left_l = float(core_lum[:, :b_strip].mean())
            right_l = float(core_lum[:, -b_strip:].mean())

            abs_bevel = abs(top_l - bot_l) + abs(left_l - right_l)
            signed_bevel = max(0.0, top_l - bot_l) + max(0.0, left_l - right_l)

            # Độ phẳng tổng hợp: ô trống có độ phẳng rất thấp (ít biến thiên, không có bevel, màu tối/ít bão hòa)
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

    # Xác định màu nền ô trống (Empty Cell Background Reference)
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
        # Trường hợp cả 64 ô gần như đầy kín, dùng màu rãnh lưới làm mốc nền
        bg_rgb = seam_rgb * 1.12

    bg_lum = float(0.299 * bg_rgb[0] + 0.587 * bg_rgb[1] + 0.114 * bg_rgb[2])
    bg_chroma = float(bg_rgb.max() - bg_rgb.min())

    # Tính điểm đặc trưng (Block Evidence Score) cho từng ô
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

    # Ngưỡng thích ứng kết hợp giới hạn vật lý (đảm bảo bàn cờ trống 100% không bị nhận nhầm)
    arr_s = np.asarray(scores, dtype=np.float32)
    s_min, s_max = float(arr_s.min()), float(arr_s.max())

    if s_max < 28.0:
        # Tất cả các ô đều là ô trống
        decision_thresh = 32.0
    elif s_min > 48.0:
        # Tất cả các ô đều đã có gạch
        decision_thresh = 30.0
    else:
        # Tìm ngưỡng phân tách tối ưu Otsu giữa cụm ô trống và cụm ô gạch
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

        # Tính độ tin cậy dựa trên khoảng cách tới ngưỡng quyết định
        margin = abs(ev - decision_thresh)
        conf = float(np.clip(0.50 + (margin / 28.0) * 0.50, 0.50, 0.999))
        cell_confidences[r][c] = round(conf, 3)
        if conf < 0.72:
            uncertain_cells.append({"row": r + 1, "col": c + 1, "confidence": round(conf, 3)})

    return board, board_mask, cell_confidences, uncertain_cells


def clean_connected_mask(raw_mask):
    """
    Lọc nhiễu hình thái học (Morphological Opening 3x3) và giữ lại các thành phần
    liên thông thuộc về khối hình chính trong slot.
    """
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


def detect_seam_pitch(profile_grad, length, min_u, max_u):
    """
    Tìm bước lưới (tile pitch u) bằng tự tương quan đạo hàm rãnh ngăn cách nội bộ.
    """
    if length < int(min_u * 1.7):
        return None, 0.0

    best_u = None
    best_response = 0.0
    g = profile_grad.astype(np.float32)
    g_mean = float(g.mean()) + 1e-5

    for u_int in range(max(8, int(round(min_u))), min(length, int(round(max_u)) + 1)):
        num_seams = int(round(length / float(u_int))) - 1
        if num_seams < 1 or num_seams > 4:
            continue
        total_ count = num_seams + 1
        step = length / float(total_count)
        seam_vals = []
        mid_vals = []
        for k in range(1, total_count):
            sx = int(round(k * step))
            l_idx = max(0, sx - 2)
            r_idx = min(length, sx + 3)
            if r_idx > l_idx:
                seam_vals.append(float(g[l_idx:r_idx].max()))
        for k in range(total_count):
            mx = int(round((k + 0.5) * step))
            l_idx = max(0, mx - 2)
            r_idx = min(length, mx + 3)
            if r_idx > l_idx:
                mid_vals.append(float(g[l_idx:r_idx].mean()))
        if seam_vals and mid_vals:
            ratio = (float(np.mean(seam_vals)) - float(np.mean(mid_vals))) / g_mean
            if ratio > best_response:
                best_response = ratio
                best_u = step

    return best_u, best_response


def solve_piece_grid_dimensions(bw_p, bh_p, unit_base, roi_lum, roi_chroma, roi_mask):
    """
    Xác định số hàng (rows) và số cột (cols) của khối hình trong khu vực chờ bằng 4 tiêu chí hình học:
    1. Tỷ lệ đồng dạng ô vuông (Squareness Constraint: u_x == u_y).
       Mọi ô đơn vị trong Block Blast đều là hình vuông 1:1. Do đó:
       - Thanh 1x4 luôn có tỷ lệ bw_p / bh_p ≈ 4.0
       - Thanh 1x5 luôn có tỷ lệ bw_p / bh_p ≈ 5.0
       Không phụ thuộc vào việc game phóng to hay thu nhỏ khối trong dock!
    2. Khớp bước rãnh ngăn cách nội bộ (Internal Seam/Groove Alignment).
    3. Quy luật tỷ lệ kích thước ô trong Dock theo số ô tối đa.
    4. Mật độ lấp đầy từng ô con (Sub-cell Occupancy Clarity).
    """
    aspect = float(bw_p) / max(1.0, float(bh_p))

    # Tính biên độ biến thiên độ sáng tại các đường rãnh dọc và ngang bên trong khối
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

            # Giới hạn kích thước vật lý hợp lệ của 1 ô trong dock (tính theo unit_base)
            if u_mean < unit_base * 0.55 or u_mean > unit_base * 1.45:
                continue

            # Tiêu chí 1: Sai số hình vuông (Squareness & Aspect Ratio Error)
            sq_err = abs(ux - uy) / max(ux, uy)
            model_aspect = float(c) / float(r)
            aspect_err = abs(aspect - model_aspect) / max(aspect, model_aspect)

            # Nếu sai lệch tỷ lệ hình vuông vượt quá 24%, loại ngay
            if sq_err > 0.24:
                continue

            # Tiêu chí 2: Sai số kích thước ô theo quy luật thu phóng của khu vực chờ
            max_dim_cells = max(r, c)
            if max_dim_cells == 5:
                # Khối 5 ô thường có kích thước ô trong khoảng [0.78 .. 1.00] * unit_base
                expected_u = float(np.clip(u_mean, unit_base * 0.80, unit_base * 0.98))
                scale_err = abs(u_mean - expected_u) / unit_base
            elif max_dim_cells == 4:
                # Khối 4 ô thường có kích thước ô trong khoảng [0.90 .. 1.06] * unit_base
                expected_u = float(np.clip(u_mean, unit_base * 0.91, unit_base * 1.05))
                scale_err = abs(u_mean - expected_u) / unit_base
            else:
                # Khối 1..3 ô thường có kích thước ô trong khoảng [0.92 .. 1.12] * unit_base
                expected_u = float(np.clip(u_mean, unit_base * 0.92, unit_base * 1.10))
                scale_err = abs(u_mean - expected_u) / unit_base

            # Tiêu chí 3: Năng lượng khớp rãnh ngăn cách giữa các ô (Seam Alignment)
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

            # Tổng chi phí (càng nhỏ càng khớp)
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
    """
    Phân tích 1 vùng chứa khối chờ xếp (slot):
    - Bóc tách hoàn toàn bóng đổ dưới đáy và viền nhiễu.
    - Xác định kích thước lưới (rows x cols) chính xác.
    - Quét từng ô con (r, c) để tạo ma trận 0/1 và tính độ tin cậy.
    """
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

    # Loại bỏ bóng đổ (bóng đổ có độ sáng tối hơn hoặc gần bằng nền và độ bão hòa màu thấp)
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

    # Tính ngưỡng tách khối khỏi nền
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

    # Xác định biên thực của khối bằng mật độ theo hàng và cột
    col_counts = mask.sum(axis=0)
    row_counts = mask.sum(axis=1)
    vx = np.where(col_counts >= max(3, int(col_counts.max() * 0.14)))[0]
    vy = np.where(row_counts >= max(3, int(row_counts.max() * 0.14)))[0]
    if len(vx) == 0 or len(vy) == 0:
        return None

    min_x, max_x = int(vx[0]), int(vx[-1])
    min_y, max_y = int(vy[0]), int(vy[-1])

    # Cắt gọt các dòng bóng đổ tối còn bám ở đáy dưới cùng của khối
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

    # Màu chuẩn của khối (mỗi khối chỉ có 1 màu đồng nhất)
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
    """
    Hàm nhận diện tổng thể Khu vực A (Bàn cờ 8x8) và Khu vực B (3 khối chờ xếp).
    Trả về đầy đủ ma trận, bitmask, độ tin cậy và danh sách cảnh báo nếu có vùng chưa rõ ràng.
    """
    img_hash = compute_image_sha256(pil_img)
    scaled_img, arr = normalize_image(pil_img, target_w=720)
    target_h, target_w, _ = arr.shape

    bx, by, bw = auto_locate_board(arr)
    cw = bw / 8.0

    board, board_mask, cell_confs, uncertain_cells = detect_board_cells(arr, bx, by, bw)

    # Định vị khu vực chờ xếp (Dock) phía dưới bàn cờ
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
                "reason": f"Không phát hiện thấy khối ở vị trí {slot_names[s]} (trống hoặc quá mờ)",
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
# 2. CÔNG THỨC ĐIỂM SCORE THỰC TẾ & BỘ GIẢI VÉT CẠN TOÀN CỤC (EXHAUSTIVE SOLVER)
# ==============================================================================
def precompute_placements(mat):
    """Tạo sẵn danh sách bitmask 64-bit cho mọi vị trí đặt hợp lệ của một ma trận khối."""
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
    """
    Tính điểm SCORE thực tế theo luật game Block Blast:
    1. Điểm đặt khối (Placement Score): +1 điểm cho mỗi ô đơn vị của khối được đặt xuống.
    2. Điểm xóa hàng/cột cơ sở (Line Clear Base Score):
       - Xóa 1 hàng/cột: 10 điểm
       - Xóa 2 hàng/cột cùng lúc: 30 điểm (10 * 2 * 3 / 2)
       - Xóa 3 hàng/cột cùng lúc: 60 điểm
       - Xóa 4 hàng/cột cùng lúc: 100 điểm
       - Xóa L hàng/cột cùng lúc: 10 * L * (L + 1) // 2
    3. Điểm thưởng chuỗi Combo (Combo Multiplier):
       - Khi xóa được hàng/cột liên tiếp, điểm xóa hàng được nhân với hệ số (current_combo_streak + 1).
    4. Điểm thưởng dọn sạch bàn cờ (All Clear Bonus): Đạt thêm thưởng nếu bàn cờ trống hoàn toàn.
    """
    placement_pts = int(cells_placed)
    if lines_cleared <= 0:
        return placement_pts, 0, 0

    new_combo = current_combo_streak + 1
    base_clear_pts = 10 * lines_cleared * (lines_cleared + 1) // 2
    clear_pts = base_clear_pts * new_combo
    total_step_score = placement_pts + clear_pts
    return total_step_score, clear_pts, new_combo


def simulate_bitboard(bmask, pmask, cells, combo_streak=0):
    """
    Mô phỏng đặt khối pmask lên bàn cờ bmask, kiểm tra xóa đồng thời cả hàng và cột,
    cập nhật trạng thái bàn cờ mới và trả về điểm SCORE thực tế.
    """
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

    step_score, clear_pts, next_combo = compute_verified_block_blast_score(
        cells, lines, combo_streak
    )
    # Thưởng dọn sạch toàn bộ bàn cờ (All-Clear)
    if clear_mask and nb == 0:
        step_score += 50

    return nb, lines, rows_cleared, cols_cleared, step_score, next_combo


def evaluate_positional_heuristic(b):
    """
    Hàm đánh giá thế cờ (Positional Heuristic) - TÁCH BIỆT HOÀN TOÀN với điểm SCORE thực tế.
    Dùng để đánh giá chất lượng bàn cờ sau khi đã đặt xong các khối:
    - Khả năng tạo lượt xóa tiếp theo (các hàng/cột đã có 6 hoặc 7 ô).
    - Không gian trống để đặt các khối lớn ở lượt tiếp theo (3x3, 2x2, 1x5, 5x1, 1x4, 4x1).
    - Hạn chế tối đa các lỗ trống 1x1 bị cô lập và sự phân mảnh bàn cờ.
    """
    empty = 64 - b.bit_count()
    e = (~b) & FULL_MASK

    # 1. Đếm không gian cho khối vuông 2x2 và 3x3
    eh2 = e & (e >> 1) & NOT_RIGHT
    space_2x2 = (eh2 & (eh2 >> 8)).bit_count()

    eh3 = eh2 & (e >> 2) & 0x3F3F3F3F3F3F3F3F
    space_3x3 = (eh3 & (eh3 >> 8) & (eh3 >> 16)).bit_count()

    # 2. Đếm hành lang cho thanh dài 1x4, 4x1, 1x5, 5x1
    eh4 = eh2 & (eh2 >> 2) & 0x0F0F0F0F0F0F0F0F
    eh5 = eh4 & (e >> 4) & 0x0707070707070707
    ev2 = e & (e >> 8)
    ev4 = ev2 & (ev2 >> 16)
    ev5 = ev4 & (e >> 32)

    bars_4 = eh4.bit_count() + ev4.bit_count()
    bars_5 = eh5.bit_count() + ev5.bit_count()

    # 3. Đếm lỗ đơn 1x1 bị vây kín 4 phía (rất khó lấp trong các lượt sau)
    up_occ = ((b << 8) & FULL_MASK) | 0xFF
    down_occ = (b >> 8) | 0xFF00000000000000
    left_occ = ((b << 1) & NOT_LEFT) | 0x0101010101010101
    right_occ = ((b >> 1) & NOT_RIGHT) | 0x8080808080808080
    isolated_1x1 = (e & up_occ & down_occ & left_occ & right_occ).bit_count()

    # 4. Đánh giá các hàng/cột gần đầy (6 hoặc 7 ô) sẵn sàng nổ ở lượt kế tiếp
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

    # 5. Độ chuyển tiếp biên (càng thấp các khối gạch càng chụm gọn về một phía)
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
    """
    Khóa xếp hạng cố định (Deterministic Lexicographical Ranking Key):
    1. Số lượng khối đặt được nhiều nhất (3 > 2 > 1).
    2. Tổng điểm SCORE thực tế (total_game_score) cao nhất.
    3. Tổng số hàng + cột xóa được (total_lines) nhiều nhất.
    4. Số hàng/cột xóa đồng thời trong 1 nước (max_simul_lines) cao nhất.
    5. Điểm đánh giá thế cờ (heuristic_score) làm tròn 4 chữ số thập phân.
    6. Quy tắc phá hòa cố định tuyệt đối theo (-slot, -row, -col) để cùng dữ liệu luôn ra đúng 1 kết quả.
    """
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
    """
    Thuật toán Tìm kiếm Vét cạn Toàn cục (100% Exhaustive Search với Bitboard Caching):
    - Kiểm tra tất cả hoán vị của các khối đang chờ.
    - Kiểm tra tất cả vị trí đặt hợp lệ của từng khối mà KHÔNG cắt nhánh sớm bằng heuristic.
    - Tối ưu hóa bước lá (khối cuối cùng) bằng bảng nhớ Exact Leaf Memoization theo
      (last_piece_slot, board_mask, combo_streak) giúp duyệt 100% tổ hợp cực nhanh và chính xác tuyệt đối.
    """
    t_start = time.perf_counter()
    if not pieces:
        if return_details:
            return {"best_steps": [], "backups": [], "stats": {"nodes": 0, "elapsed_ms": 0.0}}
        return []

    best_key = None
    best_solution = None
    top_candidates_map = {}
    nodes_evaluated = 0

    # Bảng nhớ chính xác cho khối cuối cùng: (slot_id, cur_mask, combo_streak) -> best_leaf_result
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
                # Khóa so sánh cục bộ hoàn toàn đồng bộ với khóa toàn cục
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

        # Lưu lại các phương án dự phòng tốt nhất (khác nhau về chuỗi vị trí đặt)
        sig = tuple((s["slot"], s["row"], s["col"]) for s in steps)
        prev = top_candidates_map.get(sig)
        if prev is None or rk > prev["rank_key"]:
            top_candidates_map[sig] = sol_obj

    # Duyệt toàn bộ các hoán vị theo thứ tự cố định
    all_perms = list(itertools.permutations(pieces))

    # Bảng nhớ tránh duyệt lại trạng thái trung gian trùng lặp hoàn toàn:
    # (remaining_slots_tuple, cur_mask, combo_streak, acc_score, acc_lines, max_simul)
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

            # Nếu đang ở khối cuối cùng của hoán vị và đã đặt ít nhất các khối trước đó:
            if idx == n_p - 1:
                leaf_res = evaluate_last_piece_exact(p, cur_mask, combo_streak)
                if leaf_res is not None:
                    r0, c0, nb, lines, r_clr, c_clr, step_score, next_combo, _ = leaf_res
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
                    # Khối cuối không đặt được -> ghi nhận phương án đặt một phần
                    register_solution(steps, cur_mask, acc_score, acc_lines, max_simul)
                return

            # Duyệt VÉT CẠN 100% mọi vị trí hợp lệ của khối hiện tại (không cắt nhánh!)
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
                # Nếu khối hiện tại bị kẹt, ghi nhận các bước đã đặt được và thử đặt các khối kế tiếp
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
    """Tạo lời giải thích ngắn gọn, rõ ràng vì sao phương án này được chọn."""
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

    reasons = [
        f"Đặt được {placed_cnt}/{total_pieces} khối theo thứ tự: {order_str}.",
        f"Đạt tổng điểm GAME SCORE cao nhất là +{g_score} điểm (xóa tổng cộng {t_lines} hàng/cột).",
        f"Điểm thế cờ sau lượt đi đạt {h_score} (tối ưu không gian trống và hạn chế lỗ cô lập 1x1).",
        f"Đã duyệt vét cạn {nodes_evaluated} trạng thái trong {elapsed_ms}ms.",
    ]
    return " ".join(reasons)


# ==============================================================================
# 3. ĐỊNH DẠNG BÁO CÁO KIỂM CHỨNG, THÔNG BÁO iOS & VẼ ẢNH KẾT QUẢ
# ==============================================================================
def format_matrix_str(mat):
    if not mat:
        return "(Trống)"
    return "\n".join(" ".join(str(cell) for cell in row) for row in mat)


def format_verification_report_text(board, pieces, meta):
    """
    Tạo báo cáo kiểm tra dữ liệu nhận diện (Bàn cờ 8x8 + 3 khối + Độ tin cậy)
    đúng định dạng trực quan để người dùng đối chiếu trước khi giải.
    """
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
    """
    Định dạng thông báo cho iOS Shortcut:
    - Hiển thị cảnh báo nếu độ tin cậy nhận diện thấp.
    - 3 dòng đầu là hướng dẫn thứ tự đặt khối, vị trí (Hàng, Cột) và số hàng/cột xóa.
    - Hiển thị Tổng SCORE dự kiến và bản đồ trực quan 8x8.
    """
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
    """
    Vẽ ảnh kiểm chứng nhận diện (Inspection Image):
    - Đánh số Hàng 1..8 và Cột 1..8 quanh bàn cờ.
    - Đánh dấu tâm các ô được nhận diện là đã có block (1) và các ô nghi ngờ (nếu có).
    - Đánh dấu khung bounding box và lưới chia ô (rows x cols) ngay trên 3 khối ở khu vực chờ.
    """
    img = meta["scaled_img"].copy()
    draw = ImageDraw.Draw(img, "RGBA")
    bx, by, bw, cw = meta["bx"], meta["by"], meta["bw"], meta["cw"]

    # Vẽ khung bàn cờ và số thứ tự Hàng / Cột
    draw.rectangle([bx - 2, by - 2, bx + bw + 2, by + bw + 2], outline=(0, 255, 180, 255), width=3)
    for c in range(8):
        cx = int(round(bx + (c + 0.35) * cw))
        draw.text((cx, max(4, by - 22)), f"C{c+1}", fill=(255, 255, 255, 255))
    for r in range(8):
        ry = int(round(by + (r + 0.35) * cw))
        draw.text((max(2, bx - 26), ry), f"H{r+1}", fill=(255, 255, 255, 255))

    # Vẽ trạng thái 64 ô bàn cờ
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

    # Vẽ khung kiểm chứng 3 khối ở khu vực chờ
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
    """
    Xuất ảnh PNG hướng dẫn vị trí đặt từng khối trực quan:
    - Đánh số Hàng 1..8, Cột 1..8 trên viền bàn cờ.
    - Tô màu phân biệt Bước #1 (Xanh lam), Bước #2 (Vàng), Bước #3 (Tím).
    - Ghi chú hướng dẫn đặt 3 bước và tổng điểm SCORE ngay trên ảnh.
    """
    img = meta["scaled_img"].copy()
    draw = ImageDraw.Draw(img, "RGBA")
    bx, by, bw, cw = meta["bx"], meta["by"], meta["bw"], meta["cw"]

    # Vẽ chỉ số Hàng 1..8 và Cột 1..8
    for c in range(8):
        cx = int(round(bx + (c + 0.38) * cw))
        draw.text((cx, max(4, by - 20)), f"C{c+1}", fill=(255, 255, 255, 255))
    for r in range(8):
        ry = int(round(by + (r + 0.38) * cw))
        draw.text((max(2, bx - 24), ry), f"H{r+1}", fill=(255, 255, 255, 255))

    colors = [
        (56, 189, 248, 195),   # Bước 1: Xanh Cyan
        (250, 204, 21, 195),   # Bước 2: Vàng Gold
        (192, 132, 252, 195),  # Bước 3: Tím Purple
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

    # Vẽ bảng tóm tắt thứ tự bước đi và tổng SCORE ở phía trên bàn cờ
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
    """Lưu lại dữ liệu nhận diện và kết quả giải vào file JSONL để đối chiếu giữa các lần chạy."""
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
# 5. CÁC ENDPOINT FLASK SERVER (/solve, /inspect, /solve_manual, /ui)
# ==============================================================================
@app.route("/", methods=["GET"])
def health_check():
    if request.args.get("ui") == "1":
        return render_web_inspector_ui()
    return "OK", 200


@app.route("/solve", methods=["POST"])
def solve_endpoint():
    """
    Endpoint chính tương thích 100% với Shortcut cũ và mới:
    - Mặc định trả về văn bản hướng dẫn gọn cho iOS.
    - ?format=image : Trả về ảnh PNG đánh dấu vị trí đặt khối (#1, #2, #3) và điểm SCORE.
    - ?format=verify_text : Chỉ trả về báo cáo ma trận nhận diện 8x8 và 3 khối để kiểm tra.
    - ?format=inspect_image : Trả về ảnh PNG đóng khung lưới nhận diện 8x8 và 3 khối.
    - ?format=json : Trả về toàn bộ dữ liệu JSON (ma trận, độ tin cậy, phương án tốt nhất, dự phòng, ảnh Base64).
    """
    try:
        pil_img = extract_image_from_request(request)
        if pil_img is None:
            return "⚠️ Không nhận được ảnh!", 400

        board, board_mask, pieces, meta = detect_board_and_pieces(pil_img)
        fmt = request.args.get("format", "text").lower()

        # Lưu session gần nhất để người dùng có thể mở /ui kiểm tra hoặc sửa tay ngay
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
    """
    Endpoint chuyên dụng cho Bước 1 của Chế độ 2 (Giải chính xác):
    Quét ảnh, trả về báo cáo kiểm chứng ma trận bàn cờ + 3 khối + độ tin cậy.
    """
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
    """
    Endpoint giải từ dữ liệu ma trận đã được người dùng kiểm tra hoặc chỉnh sửa thủ công.
    Nhận JSON:
    {
      "board": [[0,1,...], ...],  (8x8)
      "pieces": [
        {"slot": 1, "name": "TRÁI", "matrix": [[1,1,1,1]]},
        {"slot": 2, "name": "GIỮA", "matrix": [[1,0],[1,0],[1,1]]},
        {"slot": 3, "name": "PHẢI", "matrix": [[1,1],[1,1]]}
      ]
    }
    """
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
    """
    Giao diện Web trực quan để tải ảnh chụp màn hình lên, đối chiếu kết quả nhận diện,
    chỉnh sửa thủ công từng ô trên bàn cờ 8x8 hoặc 3 khối chờ xếp, và chạy bộ giải SCORE cao nhất.
    """
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
    <p style="color:#94a3b8;margin:4px 0;">Tải ảnh chụp màn hình lên để kiểm tra bàn cờ 8×8 và 3 khối hình. Nếu muốn chỉnh sửa, hãy chạm trực tiếp vào các ô bên dưới rồi bấm <b>Giải Lại Từ Ma Trận Đã Chỉnh Sửa</b>.</p>
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


# ==============================================================================
# 6. BỘ KIỂM THỬ NGHIÊM TÚC & ĐỐI CHIẾU VỚI BRUTE-FORCE ĐỘC LẬP (--test)
# ==============================================================================
def _create_synthetic_screenshot(
    board_mat,
    dock_mats,
    width=720,
    height=1560,
    add_shadow=True,
    piece_scales=None,
    bg_color=(28, 36, 72),
):
    """
    Tạo ảnh chụp màn hình mô phỏng thực tế với nhiều độ phân giải, tỷ lệ màn hình,
    hiệu ứng gờ nổi 3D (bevel), bóng đổ (drop shadow) và tỷ lệ co giãn khối khác nhau.
    """
    img = Image.new("RGB", (width, height), bg_color)
    draw = ImageDraw.Draw(img)

    bw = int(round(width * 0.908))
    bx = (width - bw) // 2
    by = int(round(height * 0.228))
    cw = bw / 8.0

    draw.rectangle([bx - 4, by - 4, bx + bw + 4, by + bw + 4], fill=(18, 24, 52), outline=(65, 85, 140), width=3)

    def draw_3d_tile(x0, y0, x1, y1, base_color):
        r, g, b = base_color
        top_c = (min(255, r + 48), min(255, g + 48), min(255, b + 48))
        bot_c = (max(0, r - 48), max(0, g - 48), max(0, b - 48))
        draw.rectangle([x0, y0, x1, y1], fill=base_color, outline=(14, 18, 38), width=2)
        b_w = max(2, int(round((x1 - x0) * 0.16)))
        draw.rectangle([x0 + 2, y0 + 2, x1 - 2, y0 + b_w], fill=top_c)
        draw.rectangle([x0 + 2, y1 - b_w, x1 - 2, y1 - 2], fill=bot_c)

    for r in range(8):
        for c in range(8):
            x0 = int(round(bx + c * cw))
            y0 = int(round(by + r * cw))
            x1 = int(round(bx + (c + 1) * cw))
            y1 = int(round(by + (r + 1) * cw))
            if board_mat[r][c] == 0:
                draw.rectangle([x0 + 2, y0 + 2, x1 - 2, y1 - 2], fill=(22, 29, 58), outline=(34, 44, 82))
            else:
                draw_3d_tile(x0 + 1, y0 + 1, x1 - 1, y1 - 1, (235, 88, 52))

    dock_y0 = int(round(by + bw + height * 0.042))
    dock_h = int(round(height * 0.195))
    dock_x0 = int(round(width * 0.04))
    slot_w = int(round((width * 0.92) / 3.0))
    unit_base = cw * 0.472
    piece_colors = [(52, 211, 153), (251, 191, 36), (168, 85, 247)]

    for s, mat in enumerate(dock_mats):
        if mat is None:
            continue
        rows, cols = len(mat), len(mat[0])
        if piece_scales and s < len(piece_scales) and piece_scales[s] is not None:
            u = unit_base * piece_scales[s]
        else:
            u = unit_base * 0.86 if max(rows, cols) == 5 else unit_base
        pw = cols * u
        ph = rows * u
        sx_center = dock_x0 + s * slot_w + slot_w / 2.0
        sy_center = dock_y0 + dock_h * 0.45
        px0 = sx_center - pw / 2.0
        py0 = sy_center - ph / 2.0

        if add_shadow:
            for r in range(rows):
                for c in range(cols):
                    if mat[r][c] == 1:
                        tx0 = int(round(px0 + c * u))
                        ty0 = int(round(py0 + r * u))
                        tx1 = int(round(px0 + (c + 1) * u))
                        ty1 = int(round(py0 + (r + 1) * u))
                        draw.rectangle([tx0 + 3, ty0 + 5, tx1 + 3, ty1 + 7], fill=(12, 16, 35))

        for r in range(rows):
            for c in range(cols):
                if mat[r][c] == 1:
                    tx0 = int(round(px0 + c * u))
                    ty0 = int(round(py0 + r * u))
                    tx1 = int(round(px0 + (c + 1) * u))
                    ty1 = int(round(py0 + (r + 1) * u))
                    draw_3d_tile(tx0, ty0, tx1, ty1, piece_colors[s % 3])

    return img


def brute_force_reference_solver(board_mask, pieces):
    """
    Bộ giải vét cạn thuần túy (Reference Brute-Force không dùng bất kỳ bảng nhớ/cache nào)
    để làm chuẩn đối chiếu toán học cho thuật toán chính.
    """
    best_key = None
    best_steps = []
    best_score = 0

    for perm in itertools.permutations(pieces):
        def rec(idx, cur_mask, steps, combo, acc_score, acc_lines, max_sim):
            nonlocal best_key, best_steps, best_score
            if idx == len(perm):
                if steps:
                    h = evaluate_positional_heuristic(cur_mask)
                    rk = build_solution_ranking_key(steps, acc_score, acc_lines, max_sim, h)
                    if best_key is None or rk > best_key:
                        best_key = rk
                        best_steps = [dict(s) for s in steps]
                        best_score = acc_score
                return

            p = perm[idx]
            placed = False
            for r0, c0, pmask, cells in p["placements"]:
                if (cur_mask & pmask) == 0:
                    placed = True
                    nb, lines, r_c, c_c, st_sc, nxt_c = simulate_bitboard(cur_mask, pmask, cells, combo)
                    steps.append({
                        "slot": p["slot"],
                        "name": p["name"],
                        "matrix": p["matrix"],
                        "row": r0 + 1,
                        "col": c0 + 1,
                        "lines": lines,
                        "step_score": st_sc,
                    })
                    rec(idx + 1, nb, steps, nxt_c, acc_score + st_sc, acc_lines + lines, max(max_sim, lines))
                    steps.pop()

            if not placed:
                if steps:
                    h = evaluate_positional_heuristic(cur_mask)
                    rk = build_solution_ranking_key(steps, acc_score, acc_lines, max_sim, h)
                    if best_key is None or rk > best_key:
                        best_key = rk
                        best_steps = [dict(s) for s in steps]
                        best_score = acc_score
                rec(idx + 1, cur_mask, steps, combo, acc_score, acc_lines, max_sim)

        rec(0, board_mask, [], 0, 0, 0, 0)

    return best_steps, best_score, best_key


def run_self_tests():
    print("==================================================================")
    print("BẮT ĐẦU CHẠY BỘ KIỂM THỬ NGHIÊM NGẶT BLOCK BLAST SOLVER (8 TESTS)")
    print("==================================================================")
    empty_board = [[0] * 8 for _ in range(8)]

    # TEST 1: Phân biệt thanh 1x4, 1x5, 4x1, 5x1 ở các tỷ lệ co giãn và độ phân giải khác nhau
    bar_1x4 = [[1, 1, 1, 1]]
    bar_1x5 = [[1, 1, 1, 1, 1]]
    bar_4x1 = [[1], [1], [1], [1]]
    bar_5x1 = [[1], [1], [1], [1], [1]]

    for res_w, res_h in [(750, 1334), (1170, 2532), (1080, 2400)]:
        img1 = _create_synthetic_screenshot(
            empty_board, [bar_1x4, bar_1x5, bar_4x1], width=res_w, height=res_h, piece_scales=[0.92, 0.85, 0.96]
        )
        _, _, p1, _ = detect_board_and_pieces(img1)
        assert [p["matrix"] for p in p1] == [bar_1x4, bar_1x5, bar_4x1], f"Lỗi ở độ phân giải {res_w}x{res_h}"

    img1b = _create_synthetic_screenshot(empty_board, [bar_5x1, bar_1x4, bar_1x5], piece_scales=[0.84, 0.90, 0.86])
    _, _, p1b, _ = detect_board_and_pieces(img1b)
    assert [p["matrix"] for p in p1b] == [bar_5x1, bar_1x4, bar_1x5]
    print("[PASS] Test 1: Phân biệt chính xác tuyệt đối 1x4, 1x5, 4x1, 5x1 trên màn hình 16:9, 19.5:9 và 20:9.")

    # TEST 2: Các khối không đối xứng phức tạp (Chữ L ngược, Chữ T xoay, Chữ Z, góc khuyết 3x3)
    shape_L_asym = [[1, 0], [1, 0], [1, 1]]
    shape_T_rot = [[0, 1], [1, 1], [0, 1]]
    shape_Z = [[1, 1, 0], [0, 1, 1]]
    img2 = _create_synthetic_screenshot(empty_board, [shape_L_asym, shape_T_rot, shape_Z], add_shadow=True)
    _, _, p2, meta2 = detect_board_and_pieces(img2)
    assert [p["matrix"] for p in p2] == [shape_L_asym, shape_T_rot, shape_Z]
    assert meta2["overall_confidence"] >= 0.80
    print("[PASS] Test 2: Nhận diện chính xác các khối chữ L, T, Z không đối xứng kèm đánh giá độ tin cậy.")

    # TEST 3: Bàn cờ trống hoàn toàn (0 block) và bàn cờ gần đầy (58/64 block)
    _, mask_empty, _, _ = detect_board_and_pieces(img2)
    assert mask_empty == 0, "Bàn cờ trống bị nhận nhầm là có gạch!"

    nearly_full = [[1] * 8 for _ in range(8)]
    for c in range(2, 6):
        nearly_full[4][c] = 0
    nearly_full[0][0] = 0
    img3 = _create_synthetic_screenshot(nearly_full, [bar_1x4, [[1]], [[1, 1]]])
    det_b3, _, _, _ = detect_board_and_pieces(img3)
    assert det_b3 == nearly_full, "Nhận diện sai bàn cờ gần đầy!"
    print("[PASS] Test 3: Nhận diện chuẩn xác cả bàn cờ trống 0/64 lẫn bàn cờ gần đầy 59/64 ô.")

    # TEST 4: Kiểm chứng tính bất biến (Determinism - Cùng 1 ảnh chạy nhiều lần phải ra đúng 1 kết quả)
    runs = []
    for _ in range(5):
        b_i, bm_i, pcs_i, _ = detect_board_and_pieces(img3)
        st_i = solve_best_sequence(bm_i, pcs_i)
        runs.append((b_i, [p["matrix"] for p in pcs_i], [(s["slot"], s["row"], s["col"]) for s in st_i]))
    for r_idx in range(1, 5):
        assert runs[r_idx] == runs[0], "Kết quả không đồng nhất giữa các lần chạy!"
    print("[PASS] Test 4: Đảm bảo tính bất biến 100% (cùng một ảnh luôn cho ra cùng một ma trận và phương án giải).")

    # TEST 5: Tối ưu hóa SCORE - Xóa đồng thời nhiều hàng & cột (Multi-line Clear + Combo Streak)
    # Thiết lập bàn cờ mà nếu đặt đúng vị trí sẽ xóa đồng thời 2 hàng + 1 cột (3 lines = +60 điểm xóa)
    combo_board = [[0] * 8 for _ in range(8)]
    for c in range(8):
        if c not in (3, 4):
            combo_board[2][c] = 1
            combo_board[3][c] = 1
    for r in range(8):
        if r not in (2, 3):
            combo_board[r][3] = 1
    c_mask = sum((1 << (r * 8 + c)) for r in range(8) for c in range(8) if combo_board[r][c] == 1)
    p_2x2 = {"slot": 1, "name": "TRÁI", "matrix": [[1, 1], [1, 1]], "placements": precompute_placements([[1, 1], [1, 1]])}
    p_dot = {"slot": 2, "name": "GIỮA", "matrix": [[1]], "placements": precompute_placements([[1]])}
    res5 = solve_best_sequence(c_mask, [p_dot, p_2x2], return_details=True)
    # Đặt khối 2x2 vào Hàng 3, Cột 4 (r0=2, c0=3) sẽ xóa đồng thời Hàng 3, Hàng 4 và Cột 4 (tổng 3 lines!)
    assert any(s["slot"] == 1 and s["row"] == 3 and s["col"] == 4 and s["lines"] == 3 for s in res5["best_steps"])
    assert res5["best_solution"]["game_score"] >= 65
    print(f"[PASS] Test 5: Tìm đúng nước đi xóa đồng thời 2 hàng + 1 cột đạt SCORE cao nhất (+{res5['best_solution']['game_score']} điểm).")

    # TEST 6: Đối chiếu trực tiếp với Brute-Force thuần túy để chứng minh KHÔNG bỏ sót phương án tối ưu
    test_board = [[1] * 8 for _ in range(8)]
    # Mở một số ô trống vừa đủ để tạo khoảng 15-20 vị trí đặt cho mỗi khối
    empty_coords = [
        (0, 0), (0, 1), (0, 2), (0, 3),
        (1, 0), (1, 1), (1, 2),
        (2, 0), (2, 1),
        (5, 5), (5, 6), (6, 5), (6, 6),
    ]
    for r, c in empty_coords:
        test_board[r][c] = 0
    t_mask = sum((1 << (r * 8 + c)) for r in range(8) for c in range(8) if test_board[r][c] == 1)
    test_pieces = [
        {"slot": 1, "name": "TRÁI", "matrix": [[1, 1]], "placements": precompute_placements([[1, 1]])},
        {"slot": 2, "name": "GIỮA", "matrix": [[1, 1], [1, 1]], "placements": precompute_placements([[1, 1], [1, 1]])},
        {"slot": 3, "name": "PHẢI", "matrix": [[1], [1]], "placements": precompute_placements([[1], [1]])},
    ]
    fast_details = solve_best_sequence(t_mask, test_pieces, return_details=True)
    bf_steps, bf_score, bf_key = brute_force_reference_solver(t_mask, test_pieces)

    assert fast_details["best_solution"]["game_score"] == bf_score, "Điểm SCORE khác với Brute-Force!"
    assert fast_details["best_solution"]["rank_key"] == bf_key, "Phương án khác với Brute-Force!"
    print(f"[PASS] Test 6: Khớp 100% với Brute-Force chuẩn (SCORE: +{bf_score}, duyệt {fast_details['stats']['nodes_evaluated']} nút trong {fast_details['stats']['elapsed_ms']}ms).")

    # TEST 7: Trường hợp chỉ đặt được 1 hoặc 2 khối và trường hợp hết nước đi
    tight_board = [[1] * 8 for _ in range(8)]
    tight_board[0][0] = 0
    tight_board[7][7] = 0
    tight_mask = sum((1 << (r * 8 + c)) for r in range(8) for c in range(8) if tight_board[r][c] == 1)
    p_a = {"slot": 1, "name": "TRÁI", "matrix": [[1]], "placements": precompute_placements([[1]])}
    p_b = {"slot": 2, "name": "GIỮA", "matrix": [[1, 1, 1]], "placements": precompute_placements([[1, 1, 1]])}
    p_c = {"slot": 3, "name": "PHẢI", "matrix": [[1]], "placements": precompute_placements([[1]])}
    partial_steps = solve_best_sequence(tight_mask, [p_a, p_b, p_c])
    assert len(partial_steps) == 2

    dead_steps = solve_best_sequence(tight_mask, [p_b])
    assert dead_steps == []
    print("[PASS] Test 7: Xử lý chính xác trường hợp đặt một phần (2/3 khối) và trường hợp hết nước đi hợp lệ.")

    # TEST 8: Kiểm tra báo cáo cảnh báo khi có ô chờ bị trống/mờ
    img_missing = _create_synthetic_screenshot(empty_board, [bar_1x4, None, shape_Z])
    b8, _, p8, meta8 = detect_board_and_pieces(img_missing)
    assert len(meta8["uncertain_slots"]) == 1 and meta8["uncertain_slots"][0]["slot"] == 2
    report8 = format_verification_report_text(b8, p8, meta8)
    assert "CẢNH BÁO VÙNG CHƯA RÕ RÀNG" in report8
    print("[PASS] Test 8: Tự động phát hiện và cảnh báo chính xác khi một khối ở khu vực chờ bị trống hoặc không rõ ràng.")
    print("==================================================================")
    print("HOÀN TẤT: TẤT CẢ 8 BỘ TEST KIỂM CHỨNG ĐỀU VƯỢT QUA 100%!")
    print("==================================================================")


if __name__ == "__main__":
    if "--test" in sys.argv:
        run_self_tests()
    else:
        port = int(os.environ.get("PORT", 10000))
        app.run(host="0.0.0.0", port=port)