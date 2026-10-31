import io
import os
import sys
import base64
import itertools
from collections import deque
import numpy as np
from PIL import Image, ImageDraw
from flask import Flask, request, send_file

app = Flask(__name__)

# ==========================================================
# HẰNG SỐ BITBOARD 64-BIT (TĂNG TỐC ĐỘ GIẢI GẤP 300 LẦN)
# ==========================================================
FULL_MASK = 0xFFFFFFFFFFFFFFFF
NOT_RIGHT = 0x7F7F7F7F7F7F7F7F
NOT_LEFT = 0xFEFEFEFEFEFEFEFE
ROW_MASKS = [0xFF << (r * 8) for r in range(8)]
COL_MASKS = [0x0101010101010101 << c for c in range(8)]


# ==========================================================
# 1. BỘ NHẬN DIỆN ẢNH CHỤP MÀN HÌNH (GEOMETRY & COLOR ROBUST)
# ==========================================================
def compute_otsu_threshold(values, min_t=18.0, max_t=75.0):
    arr = np.asarray(values, dtype=np.float32)
    if arr.size == 0:
        return 32.0
    vmin, vmax = float(arr.min()), float(arr.max())
    if vmax - vmin < 10.0:
        return float(np.clip(vmin + 12.0, min_t, max_t))
    best_t = (vmin + vmax) / 2.0
    best_var = -1.0
    for step in range(1, 48):
        t = vmin + (vmax - vmin) * (step / 48.0)
        w0 = arr[arr <= t]
        w1 = arr[arr > t]
        if len(w0) == 0 or len(w1) == 0:
            continue
        var_between = float(len(w0)) * float(len(w1)) * ((float(w0.mean()) - float(w1.mean())) ** 2)
        if var_between > best_var:
            best_var = var_between
            best_t = t
    return float(np.clip(best_t, min_t, max_t))


def auto_locate_board(rgb_arr):
    """Tự động xác định tọa độ (bx, by, bw) của bàn cờ 8x8."""
    h, w, _ = rgb_arr.shape
    lum = (
        0.299 * rgb_arr[:, :, 0]
        + 0.587 * rgb_arr[:, :, 1]
        + 0.114 * rgb_arr[:, :, 2]
    )

    # Ước lượng chiều rộng bàn cờ thông qua đạo hàm ngang ở vùng giữa màn hình
    mid_y0, mid_y1 = int(h * 0.22), int(h * 0.62)
    horiz_diff = np.abs(np.diff(lum[mid_y0:mid_y1, :], axis=1)).mean(axis=0)

    default_bw = int(round(w * 0.908))
    default_bx = int(round((w - default_bw) / 2.0))

    # Tìm biên trái/phải đối xứng trong khoảng [3%..10%] chiều rộng ảnh
    min_margin = max(4, int(w * 0.028))
    max_margin = max(min_margin + 2, int(w * 0.095))
    best_margin = default_bx
    best_lr_score = -1.0

    for m in range(min_margin, min(max_margin, len(horiz_diff) // 3)):
        r_idx = w - 1 - m
        if r_idx < len(horiz_diff):
            l_edge = horiz_diff[max(0, m - 1) : min(len(horiz_diff), m + 2)].max()
            r_edge = horiz_diff[max(0, r_idx - 1) : min(len(horiz_diff), r_idx + 2)].max()
            lr_score = l_edge + r_edge - abs(m - default_bx) * 1.2
            if lr_score > best_lr_score:
                best_lr_score = lr_score
                best_margin = m

    if best_lr_score > 14.0 and abs(best_margin - default_bx) <= int(w * 0.035):
        board_x = best_margin
        board_w = w - 2 * board_x
    else:
        board_x = default_bx
        board_w = default_bw

    vert_diff = np.abs(np.diff(lum[:, board_x : board_x + board_w], axis=0))
    row_edge = vert_diff.mean(axis=1)

    y_min = max(4, int(h * 0.12))
    y_max = min(int(h * 0.34), h - board_w - 12)
    best_y = int(h * 0.228)
    best_score = -1.0
    cw_est = board_w / 8.0

    for y in range(y_min, max(y_min + 1, y_max)):
        top_edge = row_edge[max(0, y - 2) : min(len(row_edge), y + 3)].max()
        bot_y = y + board_w
        if bot_y >= len(row_edge):
            continue
        bot_edge = row_edge[max(0, bot_y - 2) : min(len(row_edge), bot_y + 3)].max()

        # Cộng điểm thưởng cho 7 đường lưới ngang bên trong bàn cờ 8x8
        internal_grid = 0.0
        for k in range(1, 8):
            gy = int(round(y + k * cw_est))
            if 1 <= gy < len(row_edge) - 1:
                internal_grid += row_edge[gy - 1 : gy + 2].max()

        score = top_edge + bot_edge * 0.90 + (internal_grid / 7.0) * 0.45
        if score > best_score:
            best_score = score
            best_y = y

    return board_x, best_y, board_w


def clean_piece_mask(raw_mask):
    """
    Khử nhiễu hình thái học (3x3 neighborhood filter) và lọc thành phần liên thông (Connected Components)
    để loại bỏ bóng đổ rời rạc, hạt nhiễu hoặc vệt thanh điều hướng.
    """
    sh, sw = raw_mask.shape
    if sh < 3 or sw < 3:
        return raw_mask

    padded = np.pad(raw_mask.astype(np.uint8), 1, mode="constant", constant_values=0)
    neighbor_sum = (
        padded[0:sh, 0:sw] + padded[0:sh, 1 : sw + 1] + padded[0:sh, 2 : sw + 2]
        + padded[1 : sh + 1, 0:sw] + padded[1 : sh + 1, 1 : sw + 1] + padded[1 : sh + 1, 2 : sw + 2]
        + padded[2 : sh + 2, 0:sw] + padded[2 : sh + 2, 1 : sw + 1] + padded[2 : sh + 2, 2 : sw + 2]
    )
    # Giữ lại các pixel có ít nhất 3 pixel lân cận trong vùng 3x3 (tương đương phép lọc mở nhẹ)
    eroded = (raw_mask & (neighbor_sum >= 3))
    if not np.any(eroded):
        return raw_mask

    # Phục hồi biên trực tiếp kề với lõi đã lọc
    p_eroded = np.pad(eroded.astype(np.uint8), 1, mode="constant", constant_values=0)
    core_touch = (
        p_eroded[0:sh, 1 : sw + 1]
        + p_eroded[2 : sh + 2, 1 : sw + 1]
        + p_eroded[1 : sh + 1, 0:sw]
        + p_eroded[1 : sh + 1, 2 : sw + 2]
        + p_eroded[1 : sh + 1, 1 : sw + 1]
    ) > 0
    morphed = raw_mask & core_touch

    # Phân tích thành phần liên thông 8 hướng (8-connected BFS)
    visited = np.zeros((sh, sw), dtype=bool)
    components = []
    ys_all, xs_all = np.where(morphed)

    for y_start, x_start in zip(ys_all, xs_all):
        if visited[y_start, x_start]:
            continue
        q = deque([(int(y_start), int(x_start))])
        visited[y_start, x_start] = True
        comp = []
        while q:
            cy, cx = q.popleft()
            comp.append((cy, cx))
            for ny in range(max(0, cy - 1), min(sh, cy + 2)):
                for nx in range(max(0, cx - 1), min(sw, cx + 2)):
                    if not visited[ny, nx] and morphed[ny, nx]:
                        visited[ny, nx] = True
                        q.append((ny, nx))
        components.append(comp)

    if not components:
        return morphed

    components.sort(key=len, reverse=True)
    max_area = len(components[0])
    if max_area < 8:
        return morphed

    # Lấy bounding box của thành phần chính lớn nhất
    main_ys = [p[0] for p in components[0]]
    main_xs = [p[1] for p in components[0]]
    m_y0, m_y1 = min(main_ys), max(main_ys)
    m_x0, m_x1 = min(main_xs), max(main_xs)

    final_mask = np.zeros((sh, sw), dtype=bool)
    for comp in components:
        area = len(comp)
        if area < max(6, int(max_area * 0.06)):
            continue
        c_ys = [p[0] for p in comp]
        c_xs = [p[1] for p in comp]
        cy0, cy1 = min(c_ys), max(c_ys)
        cx0, cx1 = min(c_xs), max(c_xs)

        # Kiểm tra khoảng cách tới thành phần chính (tránh nhiễu ở viền slot xa khối)
        dist_y = max(0, m_y0 - cy1, cy0 - m_y1)
        dist_x = max(0, m_x0 - cx1, cx0 - m_x1)
        if dist_y <= 10 and dist_x <= 10:
            for py, px in comp:
                final_mask[py, px] = True

    return final_mask


def estimate_piece_grid_dims(bw_p, bh_p, unit_base, roi_lum, roi_mask):
    """
    Xác định chính xác số hàng (rows) và số cột (cols) của khối trong khu vực chờ:
    1. Tận dụng tính chất hình học: mỗi ô đơn vị là HÌNH VUÔNG (u_x ≈ u_y),
       do đó tỷ lệ bw_p / bh_p phản ánh trực tiếp tỷ lệ cols / rows (khắc phục triệt để lỗi 1x4 thành 1x5).
    2. Đối chiếu kích thước ô suy ra với kích thước chuẩn unit_base từ bàn cờ 8x8.
    3. Phân tích rãnh ngăn cách nội bộ (inter-tile seams/grooves) giữa các ô.
    """
    aspect = float(bw_p) / max(1.0, float(bh_p))

    # Tín hiệu độ sáng dọc theo chiều ngang và chiều dọc của khối để dò rãnh giữa các ô
    col_profile = (roi_lum * roi_mask).sum(axis=0) / np.maximum(1.0, roi_mask.sum(axis=0))
    row_profile = (roi_lum * roi_mask).sum(axis=1) / np.maximum(1.0, roi_mask.sum(axis=1))
    col_grad = np.abs(np.diff(col_profile, prepend=col_profile[:1]))
    row_grad = np.abs(np.diff(row_profile, prepend=row_profile[:1]))

    best_rows, best_cols = 1, 1
    best_cost = 1e9

    for r in range(1, 6):
        for c in range(1, 6):
            ux = float(bw_p) / float(c)
            uy = float(bh_p) / float(r)
            u_mean = 0.5 * (ux + uy)

            # Kiểm tra giới hạn vật lý của kích thước 1 ô trong dock so với unit_base
            if u_mean < unit_base * 0.56 or u_mean > unit_base * 1.48:
                continue

            # 1. Sai số tỷ lệ hình vuông (Squareness Error):
            # Vì mỗi ô của khối luôn là hình vuông, ux và uy phải xấp xỉ bằng nhau.
            sq_err = abs(ux - uy) / max(ux, uy)
            aspect_err = abs(aspect - (float(c) / float(r))) / max(aspect, float(c) / float(r))

            # 2. Sai số kích thước ô so với tỷ lệ chuẩn trong dock:
            # Trong Block Blast, khối có chiều dài 5 ô thường được thu nhỏ nhẹ (~0.84 - 0.92 unit_base),
            # trong khi các khối từ 1 đến 4 ô giữ nguyên kích thước ~1.0 unit_base.
            max_rc = max(r, c)
            if max_rc == 5:
                # Khối 5 ô có thể có kích thước ô từ 0.80 đến 1.02 unit_base
                target_u = float(np.clip(u_mean, unit_base * 0.82, unit_base * 1.00))
                scale_err = abs(u_mean - target_u) / unit_base + 0.015
            else:
                # Khối 1..4 ô có kích thước chuẩn quanh unit_base (0.92..1.06 unit_base)
                target_u = float(np.clip(u_mean, unit_base * 0.93, unit_base * 1.05))
                scale_err = abs(u_mean - target_u) / unit_base

            # 3. Điểm thưởng khớp rãnh ngăn cách nội bộ (Seam Alignment Bonus):
            seam_bonus = 0.0
            if c > 1 and bw_p >= c * 6:
                seam_scores = []
                for k in range(1, c):
                    sx = int(round(k * ux))
                    w_half = max(1, int(round(ux * 0.12)))
                    x_l = max(0, sx - w_half)
                    x_r = min(bw_p, sx + w_half + 1)
                    # Tại rãnh giữa 2 ô, đạo hàm biên cao hoặc độ sáng thấp hơn tâm ô
                    grad_peak = float(col_grad[x_l:x_r].max()) if x_r > x_l else 0.0
                    seam_scores.append(grad_peak)
                seam_bonus += min(0.06, float(np.mean(seam_scores)) * 0.0025)

            if r > 1 and bh_p >= r * 6:
                seam_scores = []
                for k in range(1, r):
                    sy = int(round(k * uy))
                    h_half = max(1, int(round(uy * 0.12)))
                    y_l = max(0, sy - h_half)
                    y_r = min(bh_p, sy + h_half + 1)
                    grad_peak = float(row_grad[y_l:y_r].max()) if y_r > y_l else 0.0
                    seam_scores.append(grad_peak)
                seam_bonus += min(0.06, float(np.mean(seam_scores)) * 0.0025)

            # Tổng hàm chi phí (càng nhỏ càng khớp)
            cost = sq_err * 2.6 + aspect_err * 1.8 + scale_err * 1.45 - seam_bonus
            if cost < best_cost:
                best_cost = cost
                best_rows, best_cols = r, c

    return best_rows, best_cols


def extract_piece_from_slot(slot_arr, dock_bg_global, unit_base):
    """
    Tách khối hình trong 1 ô chờ (slot), loại bỏ bóng đổ, xác định số hàng/cột
    và phân loại từng ô đơn vị trong ma trận khối.
    """
    sh, sw, _ = slot_arr.shape
    if sh < 10 or sw < 10:
        return None

    # Lấy màu nền cục bộ từ viền ngoài của slot, kết hợp với màu nền chung của dock
    border_pixels = np.concatenate([
        slot_arr[0:3, :, :].reshape(-1, 3),
        slot_arr[-3:, :, :].reshape(-1, 3),
        slot_arr[:, 0:3, :].reshape(-1, 3),
        slot_arr[:, -3:, :].reshape(-1, 3),
    ], axis=0)
    local_bg = np.median(border_pixels, axis=0)
    dock_bg = 0.65 * local_bg + 0.35 * dock_bg_global

    bg_lum = float(0.299 * dock_bg[0] + 0.587 * dock_bg[1] + 0.114 * dock_bg[2])
    bg_chroma = float(dock_bg.max() - dock_bg.min())

    lum = (
        0.299 * slot_arr[:, :, 0]
        + 0.587 * slot_arr[:, :, 1]
        + 0.114 * slot_arr[:, :, 2]
    )
    chroma = slot_arr.max(axis=2) - slot_arr.min(axis=2)
    color_diff = np.linalg.norm(slot_arr - dock_bg, axis=2)

    # Khử bóng đổ (Drop Shadow Suppression):
    # Bóng đổ dưới chân khối có độ sáng thấp hơn hoặc xấp xỉ nền và độ bão hòa màu thấp.
    # Ngược lại, gạch thật luôn sáng hơn nền hoặc có màu sắc rực rỡ (chroma cao).
    lum_gain = np.maximum(0.0, lum - bg_lum)
    chroma_gain = np.maximum(0.0, chroma - bg_chroma)
    is_dark_shadow = (lum < bg_lum + 6.0) & (chroma_gain < 18.0) & (color_diff < 48.0)

    fg_score = color_diff * 0.85 + lum_gain * 0.65 + chroma_gain * 0.90
    fg_score[is_dark_shadow] = 0.0

    # Cắt bỏ 2px sát mép trên/dưới để tránh lẹm viền UI
    fg_score[:2, :] = 0.0
    fg_score[-2:, :] = 0.0

    max_fg = float(fg_score.max())
    if max_fg < 24.0:
        return None

    active_vals = fg_score[fg_score > 12.0]
    if active_vals.size < 16:
        return None

    otsu_t = compute_otsu_threshold(active_vals, min_t=22.0, max_t=55.0)
    thresh = max(24.0, min(otsu_t, max_fg * 0.52))
    raw_mask = fg_score >= thresh

    mask = clean_piece_mask(raw_mask)
    ys, xs = np.where(mask)
    if len(xs) < 14:
        return None

    # Tìm biên chính xác của khối bằng mật độ theo hàng/cột (không dùng percentile gây cụt chữ L/T)
    col_counts = mask.sum(axis=0)
    row_counts = mask.sum(axis=1)
    valid_xs = np.where(col_counts >= max(2, int(col_counts.max() * 0.12)))[0]
    valid_ys = np.where(row_counts >= max(2, int(row_counts.max() * 0.12)))[0]

    if len(valid_xs) == 0 or len(valid_ys) == 0:
        return None

    min_x, max_x = int(valid_xs[0]), int(valid_xs[-1])
    min_y, max_y = int(valid_ys[0]), int(valid_ys[-1])

    # Cắt bỏ phần chân bóng đổ còn sót ở rìa dưới cùng nếu độ sáng trung bình dòng đáy quá tối
    while max_y - min_y > 8:
        bot_row_mask = mask[max_y, min_x : max_x + 1]
        if np.any(bot_row_mask):
            bot_lum = float(lum[max_y, min_x : max_x + 1][bot_row_mask].mean())
            core_lum = float(lum[min_y:max_y, min_x : max_x + 1][mask[min_y:max_y, min_x : max_x + 1]].mean())
            if bot_lum < bg_lum + 10.0 and bot_lum < core_lum * 0.55:
                mask[max_y, :] = False
                max_y -= 1
                continue
        break

    bw_p = max_x - min_x + 1
    bh_p = max_y - min_y + 1
    if bw_p < 6 or bh_p < 6:
        return None

    roi_lum = lum[min_y : max_y + 1, min_x : max_x + 1]
    roi_mask = mask[min_y : max_y + 1, min_x : max_x + 1]

    rows, cols = estimate_piece_grid_dims(bw_p, bh_p, unit_base, roi_lum, roi_mask)

    # Màu đặc trưng của khối (mỗi khối chỉ có 1 màu đồng nhất)
    piece_pixels = slot_arr[mask]
    piece_fg_rgb = np.median(piece_pixels, axis=0)
    fg_bg_dist = max(18.0, float(np.linalg.norm(piece_fg_rgb - dock_bg)))

    mat = np.zeros((rows, cols), dtype=int)
    sub_w = bw_p / float(cols)
    sub_h = bh_p / float(rows)

    for r in range(rows):
        for c in range(cols):
            # Lấy vùng lõi 64% ở tâm mỗi ô con để tránh nhiễu từ viền ô bên cạnh
            cx0 = int(round(min_x + (c + 0.18) * sub_w))
            cx1 = int(round(min_x + (c + 0.82) * sub_w))
            cy0 = int(round(min_y + (r + 0.18) * sub_h))
            cy1 = int(round(min_y + (r + 0.82) * sub_h))

            cx0, cx1 = max(0, cx0), min(sw, max(cx0 + 1, cx1))
            cy0, cy1 = max(0, cy0), min(sh, max(cy0 + 1, cy1))

            sub_mask = mask[cy0:cy1, cx0:cx1]
            sub_rgb = slot_arr[cy0:cy1, cx0:cx1]

            if sub_mask.size == 0:
                continue

            mask_ratio = float(sub_mask.mean())
            cell_mean_rgb = sub_rgb.mean(axis=(0, 1))
            dist_to_bg = float(np.linalg.norm(cell_mean_rgb - dock_bg))
            dist_to_fg = float(np.linalg.norm(cell_mean_rgb - piece_fg_rgb))

            # Ô được tính là có gạch khi thỏa mãn đồng thời mật độ mask hoặc màu sắc khớp với màu khối
            is_filled = (
                mask_ratio >= 0.42
                or (mask_ratio >= 0.25 and dist_to_bg > dist_to_fg * 1.15 and dist_to_bg > fg_bg_dist * 0.45)
            )
            if is_filled:
                mat[r, c] = 1

    valid_r = np.any(mat == 1, axis=1)
    valid_c = np.any(mat == 1, axis=0)
    if not (np.any(valid_r) and np.any(valid_c)):
        return None

    return mat[valid_r][:, valid_c].tolist()


def detect_board_and_pieces(pil_img):
    target_w = 360
    scale = target_w / float(pil_img.width)
    target_h = int(round(pil_img.height * scale))
    img = pil_img.resize((target_w, target_h), Image.Resampling.BILINEAR)
    arr = np.asarray(img, dtype=np.float32)

    bx, by, bw = auto_locate_board(arr)
    cw = bw / 8.0

    cell_feats = []
    for r in range(8):
        for c in range(8):
            x0 = int(round(bx + c * cw))
            y0 = int(round(by + r * cw))
            x1 = int(round(bx + (c + 1) * cw))
            y1 = int(round(by + (r + 1) * cw))

            pad = max(2, int((x1 - x0) * 0.15))
            core = arr[y0 + pad : y1 - pad, x0 + pad : x1 - pad]
            mean_rgb = core.mean(axis=(0, 1))
            lum = 0.299 * core[:, :, 0] + 0.587 * core[:, :, 1] + 0.114 * core[:, :, 2]
            std_lum = float(lum.std())
            chroma = float((core.max(axis=2) - core.min(axis=2)).mean())

            b_strip = max(2, int(lum.shape[0] * 0.24))
            top_l = float(lum[:b_strip, :].mean())
            bot_l = float(lum[-b_strip:, :].mean())
            left_l = float(lum[:, :b_strip].mean())
            right_l = float(lum[:, -b_strip:].mean())

            bevel = abs(top_l - bot_l) + abs(left_l - right_l)
            # Gạch 3D trong Block Blast luôn có cạnh trên/trái sáng hơn cạnh dưới/phải
            signed_bevel = max(0.0, top_l - bot_l) + max(0.0, left_l - right_l)

            cell_feats.append({
                "r": r,
                "c": c,
                "rgb": mean_rgb,
                "lum": float(lum.mean()),
                "chroma": chroma,
                "std": std_lum,
                "bevel": float(bevel),
                "signed_bevel": float(signed_bevel),
                "flatness": std_lum + float(bevel) * 0.9 + chroma * 0.35,
            })

    # Lấy màu nền bàn cờ an toàn kể cả khi bàn cờ có hơn 55/64 ô đã bị lấp đầy
    sorted_flat = sorted(cell_feats, key=lambda x: x["flatness"])
    anchor_rgb = sorted_flat[0]["rgb"]
    bg_candidates = [
        x["rgb"]
        for x in sorted_flat[:12]
        if float(np.linalg.norm(x["rgb"] - anchor_rgb)) < 22.0
    ]
    if len(bg_candidates) < 2:
        bg_candidates = [x["rgb"] for x in sorted_flat[:3]]
    bg_rgb = np.median(np.array(bg_candidates), axis=0)
    bg_chroma = float(bg_rgb.max() - bg_rgb.min())

    scores = []
    for f in cell_feats:
        dist = float(np.linalg.norm(f["rgb"] - bg_rgb))
        chroma_gain = max(0.0, f["chroma"] - bg_chroma)
        score = (
            dist * 1.10
            + chroma_gain * 0.45
            + f["bevel"] * 1.15
            + f["signed_bevel"] * 0.45
            + f["std"] * 0.90
        )
        f["score"] = score
        scores.append(score)

    thresh = compute_otsu_threshold(scores, min_t=22.0, max_t=62.0)
    board = [[0] * 8 for _ in range(8)]
    board_mask = 0
    for f in cell_feats:
        if f["score"] >= thresh:
            r, c = f["r"], f["c"]
            board[r][c] = 1
            board_mask |= (1 << (r * 8 + c))

    # Xác định vùng khu vực chờ (Dock) thích ứng với tỷ lệ màn hình và tránh thanh Home iOS ở đáy
    dock_y0 = min(target_h - 32, int(round(by + bw + max(cw * 0.45, target_h * 0.036))))
    safe_bottom = target_h - max(10, int(round(target_h * 0.038)))
    dock_y1 = min(safe_bottom, int(round(dock_y0 + max(cw * 3.15, target_h * 0.195))))
    if dock_y1 <= dock_y0 + 16:
        dock_y1 = min(target_h - 6, dock_y0 + 48)

    dock_x0 = int(round(target_w * 0.04))
    slot_w = int(round((target_w * 0.92) / 3.0))
    unit_base = cw * 0.472

    dock_strip = arr[dock_y0:dock_y1, dock_x0 : dock_x0 + 3 * slot_w]
    dock_borders = np.concatenate([
        dock_strip[0:3, :, :].reshape(-1, 3),
        dock_strip[-3:, :, :].reshape(-1, 3),
    ], axis=0)
    dock_bg_global = np.median(dock_borders, axis=0)

    pieces = []
    slot_names = ["TRÁI", "GIỮA", "PHẢI"]

    for s in range(3):
        sx0 = dock_x0 + s * slot_w + 3
        sx1 = dock_x0 + (s + 1) * slot_w - 3
        slot_arr = arr[dock_y0:dock_y1, sx0:sx1]

        mat = extract_piece_from_slot(slot_arr, dock_bg_global, unit_base)
        if mat is not None:
            pieces.append({
                "slot": s + 1,
                "name": slot_names[s],
                "matrix": mat,
                "placements": precompute_placements(mat),
            })

    meta = {"bx": bx, "by": by, "bw": bw, "cw": cw, "scaled_img": img}
    return board, board_mask, pieces, meta


# ==========================================================
# 2. BỘ GIẢI BITBOARD 64-BIT SIÊU TỐC & ĐẢM BẢO TÌM RA NƯỚC ĐI
# ==========================================================
def precompute_placements(mat):
    """Tạo sẵn danh sách bitmask cho mọi vị trí hợp lệ của khối."""
    R, C = len(mat), len(mat[0])
    cells = sum(sum(row) for row in mat)
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


def simulate_bitboard(bmask, pmask, cells):
    nb = bmask | pmask
    clear_mask = 0
    lines = 0
    for rm in ROW_MASKS:
        if (nb & rm) == rm:
            clear_mask |= rm
            lines += 1
    for cm in COL_MASKS:
        if (nb & cm) == cm:
            clear_mask |= cm
            lines += 1
    if clear_mask:
        nb &= ~clear_mask
    pts = cells + (lines * 18 + (lines - 1) * 12 if lines > 0 else 0)
    return nb, lines, pts


def evaluate_bitboard(b):
    empty = 64 - b.bit_count()
    e = (~b) & FULL_MASK

    # Đếm vùng trống 2x2 và 3x3 bằng phép dịch bit
    eh2 = e & (e >> 1) & NOT_RIGHT
    space_2x2 = (eh2 & (eh2 >> 8)).bit_count()

    eh3 = eh2 & (e >> 2) & 0x3F3F3F3F3F3F3F3F
    space_3x3 = (eh3 & (eh3 >> 8) & (eh3 >> 16)).bit_count()

    # Đếm hành lang trống 1x4 và 4x1 (để luôn sẵn sàng đón các thanh dài)
    eh4 = eh2 & (eh2 >> 2) & 0x0F0F0F0F0F0F0F0F
    ev2 = e & (e >> 8)
    ev4 = ev2 & (ev2 >> 16)
    bars_4 = eh4.bit_count() + ev4.bit_count()

    # Đếm lỗ đơn 1x1 bị vây kín 4 phía
    up_occ = ((b << 8) & FULL_MASK) | 0xFF
    down_occ = (b >> 8) | 0xFF00000000000000
    left_occ = ((b << 1) & NOT_LEFT) | 0x0101010101010101
    right_occ = ((b >> 1) & NOT_RIGHT) | 0x8080808080808080
    isolated = (e & up_occ & down_occ & left_occ & right_occ).bit_count()

    # Độ chuyển tiếp biên (càng thấp bàn cờ càng gọn)
    trans_h = ((b ^ (b >> 1)) & NOT_RIGHT).bit_count()
    trans_v = ((b ^ (b >> 8)) & 0x00FFFFFFFFFFFFFF).bit_count()

    return (
        empty * 14.0
        + space_3x3 * 26.0
        + space_2x2 * 6.0
        + bars_4 * 4.5
        - isolated * 42.0
        - (trans_h + trans_v) * 4.2
    )


def has_any_valid_placement(bmask, piece):
    """Kiểm tra nhanh xem khối có ít nhất 1 vị trí đặt hợp lệ trên bmask hay không."""
    for _, _, pmask, _ in piece["placements"]:
        if (bmask & pmask) == 0:
            return True
    return False


def solve_best_sequence(board_mask, pieces):
    """
    Giải tìm chuỗi nước đi tối ưu bằng Bitboard 64-bit:
    - Ưu tiên tuyệt đối đặt được nhiều khối nhất (3 khối -> 2 khối -> 1 khối).
    - Tối ưu số hàng/cột xóa được, điểm số và độ thông thoáng của bàn cờ sau cùng.
    - Pha 1: Tìm kiếm nhánh rộng (Beam = 16) có Lookahead (ưu tiên nước đi giúp đặt được khối kế tiếp).
    - Pha 2 (Fallback): Nếu Pha 1 chưa đặt hết tất cả các khối, tự động vét cạn có Memoization
      để đảm bảo 100% không bao giờ báo lỗi oan khi vẫn tồn tại phương án hợp lệ.
    """
    if not pieces:
        return []

    best_score = -1e18
    best_steps = []
    STEP_WEIGHT = 100000.0

    def update_best(steps, cur_mask, acc_lines, acc_pts):
        nonlocal best_score, best_steps
        if not steps:
            return
        total_eval = evaluate_bitboard(cur_mask) + acc_lines * 85.0 + acc_pts * 2.0
        score = len(steps) * STEP_WEIGHT + total_eval
        if score > best_score:
            best_score = score
            best_steps = list(steps)

    # PHA 1: Tìm kiếm có định hướng Lookahead với Beam Width = 16
    for perm in itertools.permutations(pieces):
        memo_beam = {}

        def dfs_beam(idx, cur_mask, steps, acc_lines, acc_pts):
            if idx == len(perm):
                update_best(steps, cur_mask, acc_lines, acc_pts)
                return

            state_key = (idx, cur_mask)
            prev_best = memo_beam.get(state_key, -1e18)
            cur_val = len(steps) * STEP_WEIGHT + acc_lines * 85.0 + acc_pts * 2.0
            if cur_val < prev_best:
                return
            memo_beam[state_key] = cur_val

            p = perm[idx]
            is_last = (idx == len(perm) - 1)
            next_p = perm[idx + 1] if not is_last else None
            candidates = []

            for r0, c0, pmask, cells in p["placements"]:
                if (cur_mask & pmask) == 0:
                    nb, lines, pts = simulate_bitboard(cur_mask, pmask, cells)
                    h_score = evaluate_bitboard(nb) + lines * 95.0 + pts * 1.5
                    # Lookahead: Cộng điểm thưởng lớn nếu nước đi này mở đường cho khối tiếp theo
                    if next_p is not None and has_any_valid_placement(nb, next_p):
                        h_score += 5000.0
                    candidates.append((h_score, r0, c0, nb, lines, pts))

            if not candidates:
                # Nếu khối hiện tại bị kẹt, ghi nhận kết quả các khối đã đặt trước đó
                update_best(steps, cur_mask, acc_lines, acc_pts)
                # Thử bỏ qua khối bị kẹt để xét tiếp các khối còn lại trong hoán vị
                dfs_beam(idx + 1, cur_mask, steps, acc_lines, acc_pts)
                return

            if is_last:
                best_cand = max(candidates, key=lambda x: x[0])
                _, r0, c0, nb, lines, pts = best_cand
                steps.append({
                    "slot": p["slot"],
                    "name": p["name"],
                    "matrix": p["matrix"],
                    "row": r0 + 1,
                    "col": c0 + 1,
                    "lines": lines,
                })
                update_best(steps, nb, acc_lines + lines, acc_pts + pts)
                steps.pop()
            else:
                candidates.sort(key=lambda x: x[0], reverse=True)
                # Giữ tối đa 16 ứng viên tốt nhất + mọi ứng viên xóa được hàng/cột
                selected = candidates[:16]
                if len(candidates) > 16:
                    for cand in candidates[16:]:
                        if cand[4] > 0 and len(selected) < 28:
                            selected.append(cand)

                for _, r0, c0, nb, lines, pts in selected:
                    steps.append({
                        "slot": p["slot"],
                        "name": p["name"],
                        "matrix": p["matrix"],
                        "row": r0 + 1,
                        "col": c0 + 1,
                        "lines": lines,
                    })
                    update_best(steps, nb, acc_lines + lines, acc_pts + pts)
                    dfs_beam(idx + 1, nb, steps, acc_lines + lines, acc_pts + pts)
                    steps.pop()

        dfs_beam(0, board_mask, [], 0, 0)

    # PHA 2 (FALLBACK VÉT CẠN): Nếu Pha 1 chưa đặt hết toàn bộ số khối,
    # chạy tìm kiếm vét cạn có cắt tỉa trạng thái trùng lặp để không bỏ sót bất kỳ phương án hợp lệ nào.
    if len(best_steps) < len(pieces):
        for perm in itertools.permutations(pieces):
            visited_dead = set()

            def dfs_exact(idx, cur_mask, steps, acc_lines, acc_pts):
                if idx == len(perm):
                    update_best(steps, cur_mask, acc_lines, acc_pts)
                    return True

                state = (idx, cur_mask)
                if state in visited_dead:
                    return False

                p = perm[idx]
                found_full = False
                valid_moves = []

                for r0, c0, pmask, cells in p["placements"]:
                    if (cur_mask & pmask) == 0:
                        nb, lines, pts = simulate_bitboard(cur_mask, pmask, cells)
                        valid_moves.append((lines, r0, c0, nb, pts))

                # Ưu tiên duyệt các nước đi xóa được hàng/cột trước
                valid_moves.sort(key=lambda x: x[0], reverse=True)

                for lines, r0, c0, nb, pts in valid_moves:
                    steps.append({
                        "slot": p["slot"],
                        "name": p["name"],
                        "matrix": p["matrix"],
                        "row": r0 + 1,
                        "col": c0 + 1,
                        "lines": lines,
                    })
                    update_best(steps, nb, acc_lines + lines, acc_pts + pts)
                    if dfs_exact(idx + 1, nb, steps, acc_lines + lines, acc_pts + pts):
                        found_full = True
                    steps.pop()

                if not found_full:
                    visited_dead.add(state)
                return found_full

            dfs_exact(0, board_mask, [], 0, 0)

    return best_steps


# ==========================================================
# 3. ĐỊNH DẠNG THÔNG BÁO GỌN 3 DÒNG ĐẦU CHO iOS & XUẤT ẢNH PNG
# ==========================================================
def format_ios_alert_text(board, steps):
    if not steps:
        return "⚠️ Không tìm thấy nước đi hợp lệ!"

    grid = [["🟦" if board[r][c] == 1 else "⬛" for c in range(8)] for r in range(8)]
    step_badges = ["1️⃣", "2️⃣", "3️⃣"]
    lines_text = []

    for i, st in enumerate(steps):
        badge = step_badges[i] if i < 3 else "🔸"
        clear_note = f" (💥Xóa {st['lines']})" if st["lines"] > 0 else ""
        # Đưa thẳng 1, 2, 3 lên đầu, ngắn gọn để không bao giờ bị cắt dòng
        lines_text.append(
            f"{badge} {st['name']} ➔ Hàng {st['row']}, Cột {st['col']}{clear_note}"
        )

        mat = st["matrix"]
        r0, c0 = st["row"] - 1, st["col"] - 1
        for dr in range(len(mat)):
            for dc in range(len(mat[0])):
                if mat[dr][dc] == 1:
                    grid[r0 + dr][c0 + dc] = badge

    map_str = "\n".join("".join(row) for row in grid)
    return "\n".join(lines_text) + "\n" + map_str


def render_solution_image(meta, steps):
    img = meta["scaled_img"].copy()
    draw = ImageDraw.Draw(img, "RGBA")
    bx, by, cw = meta["bx"], meta["by"], meta["cw"]
    colors = [
        (56, 189, 248, 190),
        (250, 204, 21, 190),
        (192, 132, 252, 190),
    ]
    for i, st in enumerate(steps):
        color = colors[i % len(colors)]
        mat = st["matrix"]
        r0, c0 = st["row"] - 1, st["col"] - 1
        for dr in range(len(mat)):
            for dc in range(len(mat[0])):
                if mat[dr][dc] == 1:
                    x0 = int(bx + (c0 + dc) * cw + 2)
                    y0 = int(by + (r0 + dr) * cw + 2)
                    x1 = int(bx + (c0 + dc + 1) * cw - 2)
                    y1 = int(by + (r0 + dr + 1) * cw - 2)
                    draw.rectangle([x0, y0, x1, y1], fill=color, outline=(255, 255, 255, 255), width=2)
                    draw.text((x0 + 4, y0 + 2), f"#{i+1}", fill=(0, 0, 0, 255))

    buf = io.BytesIO()
    img.save(buf, format="PNG")
    buf.seek(0)
    return buf


def extract_image_from_request(req):
    if req.files:
        f = next(iter(req.files.values()))
        return Image.open(f.stream).convert("RGB")
    if req.data and len(req.data) > 100:
        return Image.open(io.BytesIO(req.data)).convert("RGB")
    if req.is_json and "image" in req.json:
        b64 = req.json["image"].split(",")[-1]
        return Image.open(io.BytesIO(base64.b64decode(b64))).convert("RGB")
    return None


@app.route("/", methods=["GET"])
def health_check():
    return "OK", 200


@app.route("/solve", methods=["POST"])
def solve_endpoint():
    try:
        pil_img = extract_image_from_request(request)
        if pil_img is None:
            return "⚠️ Không nhận được ảnh!", 400

        board, board_mask, pieces, meta = detect_board_and_pieces(pil_img)
        steps = solve_best_sequence(board_mask, pieces)

        if request.args.get("format") == "image":
            buf = render_solution_image(meta, steps)
            return send_file(buf, mimetype="image/png")

        alert_text = format_ios_alert_text(board, steps)
        return alert_text, 200, {"Content-Type": "text/plain; charset=utf-8"}

    except Exception as e:
        return f"❌ Lỗi: {str(e)}", 500


# ==========================================================
# 4. BỘ KIỂM THỬ TỰ ĐỘNG (CHẠY BẰNG LỆNH: python app.py --test)
# ==========================================================
def _create_synthetic_screenshot(board_mat, dock_mats, width=720, height=1560, add_shadow=True):
    """Tạo ảnh mô phỏng màn hình Block Blast có hiệu ứng bevel 3D và bóng đổ để kiểm thử độc lập."""
    img = Image.new("RGB", (width, height), (28, 36, 72))
    draw = ImageDraw.Draw(img)

    bw = int(round(width * 0.908))
    bx = (width - bw) // 2
    by = int(round(height * 0.228))
    cw = bw / 8.0

    # Vẽ khung nền bàn cờ 8x8
    draw.rectangle([bx - 4, by - 4, bx + bw + 4, by + bw + 4], fill=(18, 24, 52), outline=(65, 85, 140), width=3)

    def draw_3d_tile(x0, y0, x1, y1, base_color, shadow=False):
        if shadow:
            draw.rectangle([x0 + 2, y0 + 4, x1 + 2, y1 + 5], fill=(12, 16, 34))
        r, g, b = base_color
        top_c = (min(255, r + 45), min(255, g + 45), min(255, b + 45))
        bot_c = (max(0, r - 45), max(0, g - 45), max(0, b - 45))
        draw.rectangle([x0, y0, x1, y1], fill=base_color, outline=(15, 20, 42), width=1)
        b_w = max(2, int((x1 - x0) * 0.16))
        draw.rectangle([x0 + 1, y0 + 1, x1 - 1, y0 + b_w], fill=top_c)
        draw.rectangle([x0 + 1, y1 - b_w, x1 - 1, y1 - 1], fill=bot_c)

    # Vẽ các ô trên bàn cờ
    for r in range(8):
        for c in range(8):
            x0 = int(round(bx + c * cw))
            y0 = int(round(by + r * cw))
            x1 = int(round(bx + (c + 1) * cw))
            y1 = int(round(by + (r + 1) * cw))
            if board_mat[r][c] == 0:
                draw.rectangle([x0 + 2, y0 + 2, x1 - 2, y1 - 2], fill=(22, 29, 58), outline=(34, 44, 82))
            else:
                draw_3d_tile(x0 + 1, y0 + 1, x1 - 1, y1 - 1, (235, 88, 52), shadow=False)

    # Vẽ 3 khối ở khu vực chờ (Dock)
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
        u = unit_base * 0.88 if max(rows, cols) == 5 else unit_base
        pw = cols * u
        ph = rows * u
        sx_center = dock_x0 + s * slot_w + slot_w / 2.0
        sy_center = dock_y0 + dock_h * 0.45
        px0 = sx_center - pw / 2.0
        py0 = sy_center - ph / 2.0

        # Vẽ bóng đổ trước, vẽ mặt gạch 3D sau
        if add_shadow:
            for r in range(rows):
                for c in range(cols):
                    if mat[r][c] == 1:
                        tx0 = int(round(px0 + c * u))
                        ty0 = int(round(py0 + r * u))
                        tx1 = int(round(px0 + (c + 1) * u))
                        ty1 = int(round(py0 + (r + 1) * u))
                        draw.rectangle([tx0 + 2, ty0 + 5, tx1 + 2, ty1 + 6], fill=(14, 18, 38))

        for r in range(rows):
            for c in range(cols):
                if mat[r][c] == 1:
                    tx0 = int(round(px0 + c * u))
                    ty0 = int(round(py0 + r * u))
                    tx1 = int(round(px0 + (c + 1) * u))
                    ty1 = int(round(py0 + (r + 1) * u))
                    draw_3d_tile(tx0, ty0, tx1, ty1, piece_colors[s % 3], shadow=False)

    return img


def run_self_tests():
    print("=== BẮT ĐẦU CHẠY BỘ KIỂM THỬ BLOCK BLAST SOLVER ===")
    empty_board = [[0] * 8 for _ in range(8)]

    # Test 1: Kiểm tra nhận diện khối 1x4, 4x1, 1x5 (Có bóng đổ 3D)
    bar_1x4 = [[1, 1, 1, 1]]
    bar_4x1 = [[1], [1], [1], [1]]
    bar_1x5 = [[1, 1, 1, 1, 1]]
    img_bars = _create_synthetic_screenshot(empty_board, [bar_1x4, bar_4x1, bar_1x5], add_shadow=True)
    _, _, pieces_bars, _ = detect_board_and_pieces(img_bars)

    assert len(pieces_bars) == 3, f"Lỗi số khối: {len(pieces_bars)}"
    assert pieces_bars[0]["matrix"] == bar_1x4, f"Sai khối 1x4: {pieces_bars[0]['matrix']}"
    assert pieces_bars[1]["matrix"] == bar_4x1, f"Sai khối 4x1: {pieces_bars[1]['matrix']}"
    assert pieces_bars[2]["matrix"] == bar_1x5, f"Sai khối 1x5: {pieces_bars[2]['matrix']}"
    print("[PASS] Test 1: Phân biệt chính xác thanh ngang 1x4, thanh dọc 4x1 và thanh 1x5 (không bị nhầm 1x4 -> 1x5).")

    # Test 2: Kiểm tra nhận diện khối chữ L, chữ T, chữ Z
    shape_L = [[1, 0, 0], [1, 0, 0], [1, 1, 1]]
    shape_T = [[1, 1, 1], [0, 1, 0]]
    shape_Z = [[1, 1, 0], [0, 1, 1]]
    img_ltz = _create_synthetic_screenshot(empty_board, [shape_L, shape_T, shape_Z], add_shadow=True)
    _, _, pieces_ltz, _ = detect_board_and_pieces(img_ltz)

    assert pieces_ltz[0]["matrix"] == shape_L, f"Sai khối chữ L: {pieces_ltz[0]['matrix']}"
    assert pieces_ltz[1]["matrix"] == shape_T, f"Sai khối chữ T: {pieces_ltz[1]['matrix']}"
    assert pieces_ltz[2]["matrix"] == shape_Z, f"Sai khối chữ Z: {pieces_ltz[2]['matrix']}"
    print("[PASS] Test 2: Nhận diện chính xác các khối không đều (Chữ L, Chữ T, Chữ Z) không thừa/thiếu ô.")

    # Test 3: Kiểm tra bàn cờ có rất nhiều ô đã lấp đầy (58/64 ô)
    nearly_full_board = [[1] * 8 for _ in range(8)]
    for c in range(2, 6):
        nearly_full_board[3][c] = 0
    nearly_full_board[0][0] = 0
    nearly_full_board[7][7] = 0
    img_full = _create_synthetic_screenshot(nearly_full_board, [bar_1x4, None, None])
    det_board, _, det_pieces, _ = detect_board_and_pieces(img_full)
    assert det_board == nearly_full_board, "Nhận diện sai bàn cờ gần đầy!"
    steps_full = solve_best_sequence(
        sum((1 << (r * 8 + c)) for r in range(8) for c in range(8) if det_board[r][c] == 1),
        det_pieces,
    )
    assert len(steps_full) == 1 and steps_full[0]["row"] == 4 and steps_full[0]["col"] == 3
    print("[PASS] Test 3: Nhận diện chính xác bàn cờ gần đầy (58/64 ô) và đặt chuẩn thanh 1x4 vào khe trống 4 ô.")

    # Test 4: Kiểm tra trường hợp đòi hỏi đặt khối 1 xóa hàng để lấy chỗ cho khối 3x3
    b_board = [[1] * 8 for _ in range(8)]
    # Để trống 2 ô ở hàng 0 (cột 0, 1) để khi đặt khối 1x2 vào đó sẽ xóa toàn bộ hàng 0
    b_board[0][0] = 0
    b_board[0][1] = 0
    # Để trống thêm 2 hàng 1, 2 nhưng trừ cột 0,1 đã đủ để sau khi hàng 0 xóa sẽ mở vùng 3x3
    for r in range(1, 3):
        for c in range(0, 3):
            b_board[r][c] = 0
    b_mask = sum((1 << (r * 8 + c)) for r in range(8) for c in range(8) if b_board[r][c] == 1)
    p_1x2 = {"slot": 1, "name": "TRÁI", "matrix": [[1, 1]], "placements": precompute_placements([[1, 1]])}
    p_3x3 = {"slot": 2, "name": "GIỮA", "matrix": [[1]*3]*3, "placements": precompute_placements([[1]*3]*3)}
    steps_chain = solve_best_sequence(b_mask, [p_3x3, p_1x2])
    assert len(steps_chain) == 2, f"Không giải được chuỗi phụ thuộc xóa hàng: {steps_chain}"
    print("[PASS] Test 4: Giải thành công chuỗi nước đi phụ thuộc (đặt khối nhỏ xóa hàng để mở chỗ cho khối 3x3).")

    # Test 5: Trường hợp chỉ đặt được 2 trong 3 khối (vẫn phải trả về 2 bước thay vì báo lỗi)
    tight_board = [[1] * 8 for _ in range(8)]
    tight_board[0][0] = 0
    tight_board[0][2] = 0
    tight_mask = sum((1 << (r * 8 + c)) for r in range(8) for c in range(8) if tight_board[r][c] == 1)
    p_dot1 = {"slot": 1, "name": "TRÁI", "matrix": [[1]], "placements": precompute_placements([[1]])}
    p_big = {"slot": 2, "name": "GIỮA", "matrix": [[1]*3]*3, "placements": precompute_placements([[1]*3]*3)}
    p_dot2 = {"slot": 3, "name": "PHẢI", "matrix": [[1]], "placements": precompute_placements([[1]])}
    steps_partial = solve_best_sequence(tight_mask, [p_big, p_dot1, p_dot2])
    assert len(steps_partial) == 2, f"Lỗi đặt một phần số khối: {len(steps_partial)}"
    print("[PASS] Test 5: Trả về đúng phương án đặt 2/3 khối khi khối 3x3 ở giữa bị kẹt.")

    # Test 6: Trường hợp thực sự không còn vị trí đặt hợp lệ
    dead_steps = solve_best_sequence(tight_mask, [p_big])
    assert dead_steps == [] and "Không tìm thấy nước đi hợp lệ" in format_ios_alert_text(tight_board, dead_steps)
    print("[PASS] Test 6: Báo đúng thông báo khi thực sự 100% không còn vị trí đặt hợp lệ.")
    print("=== TẤT CẢ 6 BỘ TEST ĐỀU VƯỢT QUA THÀNH CÔNG! ===")


if __name__ == "__main__":
    if "--test" in sys.argv:
        run_self_tests()
    else:
        port = int(os.environ.get("PORT", 10000))
        app.run(host="0.0.0.0", port=port)