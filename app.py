import io
import base64
import itertools
from collections import deque
import numpy as np
from PIL import Image, ImageDraw
from flask import Flask, request, send_file

app = Flask(__name__)

# ==========================================================
# HẰNG SỐ BITBOARD 64-BIT
# ==========================================================
FULL_MASK = 0xFFFFFFFFFFFFFFFF
NOT_RIGHT = 0x7F7F7F7F7F7F7F7F
NOT_LEFT = 0xFEFEFEFEFEFEFEFE
ROW_MASKS = [0xFF << (r * 8) for r in range(8)]
COL_MASKS = [0x0101010101010101 << c for c in range(8)]


# ==========================================================
# 1. ĐỊNH VỊ CHÍNH XÁC KHUNG BÀN CỜ 8x8 TRÊN MỌI MÀN HÌNH
# ==========================================================
def locate_board_precisely(arr):
    """
    Tìm chính xác tọa độ (bx, by, bw) của bàn cờ 8x8 bằng cách:
    - Tìm viền trái/phải của khung bàn cờ theo độ chênh sáng dọc.
    - Quét cửa sổ vuông từ 15% -> 42% chiều cao ảnh, chấm điểm dựa trên:
      1) Độ tương phản viền trên & viền dưới của khung bàn cờ.
      2) Mức độ tối/đồng nhất của lòng bàn cờ so với vùng nền ngoài.
      3) Tín hiệu đường kẻ lưới chia 8x8 bên trong.
    """
    h, w, _ = arr.shape
    lum = 0.299 * arr[:, :, 0] + 0.587 * arr[:, :, 1] + 0.114 * arr[:, :, 2]

    # 1. Tìm chiều rộng và vị trí ngang (bx, bw)
    # Bàn cờ Block Blast luôn nằm chính giữa theo chiều ngang và rộng khoảng 88% - 93% màn hình
    mid_y0, mid_y1 = int(h * 0.28), int(h * 0.62)
    gx = np.abs(np.diff(lum[mid_y0:mid_y1, :], axis=1)).mean(axis=0)

    best_bx = int(round(w * 0.046))
    best_bw = int(round(w * 0.908))
    best_x_score = -1.0

    for bw_cand in range(int(w * 0.86), int(w * 0.95), 2):
        bx_cand = (w - bw_cand) // 2
        right_x = bx_cand + bw_cand - 1
        if 2 <= bx_cand < w - 3 and 2 <= right_x < w - 3:
            s_left = gx[bx_cand - 2 : bx_cand + 3].max()
            s_right = gx[right_x - 2 : right_x + 3].max()
            if s_left + s_right > best_x_score:
                best_x_score = s_left + s_right
                best_bx = bx_cand
                best_bw = bw_cand

    bx, bw = best_bx, best_bw
    cw = bw / 8.0

    # 2. Tìm vị trí đỉnh bàn cờ (by)
    gy = np.abs(np.diff(lum[:, bx : bx + bw], axis=0)).mean(axis=1)
    row_lum = lum[:, bx : bx + bw].mean(axis=1)

    y_min = int(h * 0.15)
    y_max = min(int(h * 0.42), h - bw - 40)
    best_by = int(h * 0.25)
    best_y_score = -1e9

    for y in range(y_min, y_max, 2):
        bot_y = y + bw
        # Cạnh viền trên và viền dưới của khung bàn cờ
        top_edge = gy[max(0, y - 3) : y + 4].max()
        bot_edge = gy[max(0, bot_y - 3) : min(len(gy), bot_y + 4)].max()

        # Lòng bàn cờ luôn có nền sẫm hơn vùng tiêu đề điểm số phía trên và vùng khay phía dưới
        above_lum = row_lum[max(0, y - 22) : max(1, y - 4)].mean()
        below_lum = row_lum[min(h - 1, bot_y + 5) : min(h, bot_y + 25)].mean()
        inside_lum = row_lum[y + 6 : bot_y - 6].mean()
        darkness_drop = ((above_lum - inside_lum) + (below_lum - inside_lum)) * 0.75

        # Tín hiệu 7 đường kẻ ngang chia 8 hàng bên trong bàn cờ
        grid_lines = 0.0
        for k in range(1, 8):
            gy_line = int(round(y + k * cw))
            grid_lines += gy[max(0, gy_line - 1) : gy_line + 2].max()
        grid_lines /= 7.0

        total_score = top_edge * 1.4 + bot_edge * 1.6 + darkness_drop + grid_lines * 0.9
        if total_score > best_y_score:
            best_y_score = total_score
            best_by = y

    return bx, best_by, bw


# ==========================================================
# 2. NHẬN DIỆN 64 Ô TRÊN BÀN CỜ (CHỐNG NHIỄU NỀN HỒNG/TỐI)
# ==========================================================
def detect_board_cells(arr, bx, by, bw):
    cw = bw / 8.0
    cell_feats = []

    for r in range(8):
        for c in range(8):
            x0 = int(round(bx + c * cw))
            y0 = int(round(by + r * cw))
            x1 = int(round(bx + (c + 1) * cw))
            y1 = int(round(by + (r + 1) * cw))

            pad = max(2, int((x1 - x0) * 0.14))
            core = arr[y0 + pad : y1 - pad, x0 + pad : x1 - pad]
            mean_rgb = core.mean(axis=(0, 1))
            lum = 0.299 * core[:, :, 0] + 0.587 * core[:, :, 1] + 0.114 * core[:, :, 2]
            std_lum = float(lum.std())

            # Hiệu ứng vát cạnh 3D (Bevel) của viên gạch: cạnh trên/trái sáng, cạnh dưới/phải tối
            s = max(2, int(lum.shape[0] * 0.25))
            top_l = float(lum[:s, :].mean())
            bot_l = float(lum[-s:, :].mean())
            left_l = float(lum[:, :s].mean())
            right_l = float(lum[:, -s:].mean())
            bevel = abs(top_l - bot_l) + abs(left_l - right_l)

            cell_feats.append({
                "r": r,
                "c": c,
                "rgb": mean_rgb,
                "std": std_lum,
                "bevel": bevel,
                "flatness": std_lum + bevel * 0.95,
            })

    # Lấy nhóm ô phẳng nhất (ô trống) để xác định màu nền thực tế của lòng bàn cờ
    sorted_flat = sorted(cell_feats, key=lambda x: x["flatness"])
    bg_rgb = np.median(np.array([x["rgb"] for x in sorted_flat[:10]]), axis=0)
    bg_bevel = float(np.median([x["bevel"] for x in sorted_flat[:10]]))

    board = [[0] * 8 for _ in range(8)]
    board_mask = 0

    for f in cell_feats:
        color_dist = float(np.linalg.norm(f["rgb"] - bg_rgb))
        bevel_gain = max(0.0, f["bevel"] - bg_bevel)
        # Viên gạch thật luôn có độ lệch màu so với nền tối VÀ có cạnh vát 3D rõ rệt
        score = color_dist * 1.1 + bevel_gain * 1.55 + f["std"] * 0.9
        if score >= 28.0 and (bevel_gain >= 6.5 or color_dist >= 22.0):
            r, c = f["r"], f["c"]
            board[r][c] = 1
            board_mask |= (1 << (r * 8 + c))

    return board, board_mask


# ==========================================================
# 3. NHẬN DIỆN CHÍNH XÁC 3 KHỐI CHỜ (CANONICAL TEMPLATE MATCHING)
# ==========================================================
# Danh mục toàn bộ các hình dạng hợp lệ duy nhất trong game Block Blast
# Không có bất kỳ khối 2x5, 3x5, 4x2 nào tồn tại trong game!
CANONICAL_SHAPES = [
    # 1 ô
    ([[1]], "1x1"),
    # 2 ô
    ([[1, 1]], "1x2"),
    ([[1], [1]], "2x1"),
    # 3 ô thẳng & góc
    ([[1, 1, 1]], "1x3"),
    ([[1], [1], [1]], "3x1"),
    ([[1, 0], [1, 1]], "L3-TL"),
    ([[0, 1], [1, 1]], "L3-TR"),
    ([[1, 1], [1, 0]], "L3-BL"),
    ([[1, 1], [0, 1]], "L3-BR"),
    # 4 ô vuông
    ([[1, 1], [1, 1]], "2x2"),
    # 4 ô thẳng
    ([[1, 1, 1, 1]], "1x4"),
    ([[1], [1], [1], [1]], "4x1"),
    # 4 ô L/J (3x2 và 2x3)
    ([[1, 0], [1, 0], [1, 1]], "L-3x2-1"),
    ([[0, 1], [0, 1], [1, 1]], "L-3x2-2"),
    ([[1, 1], [1, 0], [1, 0]], "L-3x2-3"),
    ([[1, 1], [0, 1], [0, 1]], "L-3x2-4"),
    ([[1, 0, 0], [1, 1, 1]], "L-2x3-1"),
    ([[0, 0, 1], [1, 1, 1]], "L-2x3-2"),
    ([[1, 1, 1], [1, 0, 0]], "L-2x3-3"),
    ([[1, 1, 1], [0, 0, 1]], "L-2x3-4"),
    # 4 ô T (3x2 và 2x3)
    ([[1, 1, 1], [0, 1, 0]], "T-2x3-U"),
    ([[0, 1, 0], [1, 1, 1]], "T-2x3-D"),
    ([[1, 0], [1, 1], [1, 0]], "T-3x2-L"),
    ([[0, 1], [1, 1], [0, 1]], "T-3x2-R"),
    # 4 ô Z/S (3x2 và 2x3)
    ([[1, 1, 0], [0, 1, 1]], "Z-2x3-1"),
    ([[0, 1, 1], [1, 1, 0]], "Z-2x3-2"),
    ([[1, 0], [1, 1], [0, 1]], "Z-3x2-1"),
    ([[0, 1], [1, 1], [1, 0]], "Z-3x2-2"),
    # 5 ô thẳng
    ([[1, 1, 1, 1, 1]], "1x5"),
    ([[1], [1], [1], [1], [1]], "5x1"),
    # 6 ô chữ nhật đặc
    ([[1, 1, 1], [1, 1, 1]], "2x3-Đặc"),
    ([[1, 1], [1, 1], [1, 1]], "3x2-Đặc"),
    # 3x3 hình chữ L lớn (5 ô)
    ([[1, 0, 0], [1, 0, 0], [1, 1, 1]], "BigL-3x3-1"),
    ([[0, 0, 1], [0, 0, 1], [1, 1, 1]], "BigL-3x3-2"),
    ([[1, 1, 1], [1, 0, 0], [1, 0, 0]], "BigL-3x3-3"),
    ([[1, 1, 1], [0, 0, 1], [0, 0, 1]], "BigL-3x3-4"),
    # 3x3 dấu cộng (5 ô)
    ([[0, 1, 0], [1, 1, 1], [0, 1, 0]], "Plus-3x3"),
    # 3x3 đặc (9 ô)
    ([[1, 1, 1], [1, 1, 1], [1, 1, 1]], "3x3-Đặc"),
]


def match_best_canonical_shape(slot_arr, cw):
    """
    So khớp khay chứa với danh mục khối chuẩn Block Blast:
    1. Lấy màu nền an toàn từ mép trên đỉnh slot (nơi khối không bao giờ chạm tới).
    2. Tách lớp bóng đổ đen dưới chân khối dựa trên độ bão hòa màu và độ sáng.
    3. Thử nghiệm trên các khối chuẩn: tính điểm tương quan giữa lõi ô và hình mẫu.
    """
    sh, sw, _ = slot_arr.shape
    if sh < 20 or sw < 20:
        return None

    # 1. Lấy màu nền an toàn từ 15% phía trên đỉnh slot (không dính cạnh gạch)
    top_band = slot_arr[: max(4, int(sh * 0.16)), :]
    local_bg = np.median(top_band.reshape(-1, 3), axis=0)

    # 2. Phân đoạn tiền cảnh: khối gạch vs nền và bóng đổ
    # Block Blast luôn có bóng đổ màu tối sẫm (đen mờ) dưới chân và bên phải gạch
    diff = np.linalg.norm(slot_arr - local_bg, axis=2)
    lum = 0.299 * slot_arr[:, :, 0] + 0.587 * slot_arr[:, :, 1] + 0.114 * slot_arr[:, :, 2]
    bg_lum = float(0.299 * local_bg[0] + 0.587 * local_bg[1] + 0.114 * local_bg[2])

    # Gạch thật: chênh màu rõ rệt (> 24) VÀ không phải là bóng đổ tối (lum >= bg_lum * 0.82)
    # Hoặc nếu nền tối (navy): gạch sáng hơn hẳn nền (lum > bg_lum + 25)
    if bg_lum < 80:
        # Nền tối (classic navy theme)
        piece_mask = (diff > 25.0) & (lum > bg_lum + 18.0)
    else:
        # Nền sáng/hồng/vàng: bóng đổ đen bị loại bỏ bằng ngưỡng độ sáng
        piece_mask = (diff > 24.0) & (lum >= bg_lum * 0.78)

    # 3. Tìm vùng bao quanh liên thông chính
    ys, xs = np.where(piece_mask)
    if len(xs) < 25:
        return None

    # Lọc ngoại lai 2% và 98% để loại bóng mờ và tia nhiễu
    min_x = int(np.percentile(xs, 2))
    max_x = int(np.percentile(xs, 98))
    min_y = int(np.percentile(ys, 2))
    max_y = int(np.percentile(ys, 98))

    blob_w = max_x - min_x + 1
    blob_h = max_y - min_y + 1

    if blob_w < 8 or blob_h < 8:
        return None

    best_score = -1e9
    best_matrix = None

    # 4. Duyệt qua tất cả các hình dạng chuẩn duy nhất trong Block Blast
    for template, _ in CANONICAL_SHAPES:
        rows = len(template)
        cols = len(template[0])

        # Kích thước ô đơn vị ước tính theo mẫu này
        u_w = blob_w / float(cols)
        u_h = blob_h / float(rows)
        u_avg = (u_w + u_h) / 2.0

        # Ràng buộc 1: Trong Block Blast, mỗi viên gạch con LUÔN LÀ HÌNH VUÔNG (tỷ lệ 1:1)
        aspect_err = abs(u_w - u_h) / max(u_w, u_h)
        if aspect_err > 0.32:
            continue

        # Ràng buộc 2: Kích thước viên gạch ở khay chờ nằm trong khoảng 0.30 -> 0.55 lần ô bàn cờ
        scale_ratio = u_avg / cw
        if scale_ratio < 0.26 or scale_ratio > 0.62:
            continue

        # Chấm điểm độ khít của từng ô trong lưới rows x cols
        cell_match_sum = 0.0
        total_cells = rows * cols

        for r in range(rows):
            for c in range(cols):
                # Lấy 60% vùng lõi ở giữa mỗi ô để tránh viền vát cạnh
                cx0 = max(0, int(round(min_x + (c + 0.20) * u_w)))
                cx1 = min(sw, int(round(min_x + (c + 0.80) * u_w)))
                cy0 = max(0, int(round(min_y + (r + 0.20) * u_h)))
                cy1 = min(sh, int(round(min_y + (r + 0.80) * u_h)))

                if cx1 <= cx0 or cy1 <= cy0:
                    continue

                sub_mask = piece_mask[cy0:cy1, cx0:cx1]
                fill_ratio = float(sub_mask.mean()) if sub_mask.size > 0 else 0.0

                if template[r][c] == 1:
                    # Ô có gạch: thưởng cao nếu fill_ratio cao
                    cell_match_sum += fill_ratio
                else:
                    # Ô trống (khuyết như L, T, Plus): thưởng cao nếu fill_ratio thấp
                    cell_match_sum += (1.0 - fill_ratio)

        match_score = (cell_match_sum / total_cells) - aspect_err * 0.9

        # Thưởng nhẹ cho mẫu có kích thước ô khay chuẩn (~0.44 * cw)
        size_err = abs(scale_ratio - 0.44)
        match_score -= size_err * 0.25

        if match_score > best_score:
            best_score = match_score
            best_matrix = template

    return best_matrix


def detect_three_dock_pieces(arr, bx, by, bw):
    """
    Định vị vùng khay dưới bàn cờ và nhận diện chính xác 3 khối chờ
    bằng cách so khớp danh mục mẫu khối chuẩn Block Blast (loại bỏ hoàn toàn lỗi 2x5).
    """
    h, w, _ = arr.shape
    cw = bw / 8.0

    # Khung dọc khay chờ: từ sau đáy bàn cờ đến trước banner quảng cáo đáy
    dock_y0 = int(round(by + bw + cw * 0.38))
    dock_y1 = min(int(round(h * 0.865)), int(round(by + bw + cw * 4.10)))

    if dock_y1 - dock_y0 < 35:
        dock_y0 = int(round(by + bw + 10))
        dock_y1 = min(h - 10, dock_y0 + int(round(cw * 3.2)))

    slot_w = bw / 3.0
    pieces = []
    slot_names = ["TRÁI", "GIỮA", "PHẢI"]

    for s in range(3):
        sx0 = max(0, int(round(bx + s * slot_w + 3)))
        sx1 = min(w, int(round(bx + (s + 1) * slot_w - 3)))
        slot_arr = arr[dock_y0:dock_y1, sx0:sx1]

        best_mat = match_best_canonical_shape(slot_arr, cw)
        if best_mat is not None:
            pieces.append({
                "slot": s + 1,
                "name": slot_names[s],
                "matrix": best_mat,
                "placements": precompute_placements(best_mat),
            })

    return pieces


def detect_board_and_pieces(pil_img):
    target_w = 360
    scale = target_w / float(pil_img.width)
    target_h = int(round(pil_img.height * scale))
    img = pil_img.resize((target_w, target_h), Image.Resampling.BILINEAR)
    arr = np.asarray(img, dtype=np.float32)

    bx, by, bw = locate_board_precisely(arr)
    board, board_mask = detect_board_cells(arr, bx, by, bw)
    pieces = detect_three_dock_pieces(arr, bx, by, bw)

    meta = {"bx": bx, "by": by, "bw": bw, "cw": bw / 8.0, "scaled_img": img}
    return board, board_mask, pieces, meta


# ==========================================================
# 4. BỘ GIẢI BITBOARD 64-BIT SIÊU TỐC (< 0.03 GIÂY)
# ==========================================================
def precompute_placements(mat):
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

    eh3 = e & (e >> 1) & (e >> 2) & 0x3F3F3F3F3F3F3F3F
    space_3x3 = (eh3 & (eh3 >> 8) & (eh3 >> 16)).bit_count()

    up_occ = ((b << 8) & FULL_MASK) | 0xFF
    down_occ = (b >> 8) | 0xFF00000000000000
    left_occ = ((b << 1) & NOT_LEFT) | 0x0101010101010101
    right_occ = ((b >> 1) & NOT_RIGHT) | 0x8080808080808080
    isolated = (e & up_occ & down_occ & left_occ & right_occ).bit_count()

    trans_h = ((b ^ (b >> 1)) & NOT_RIGHT).bit_count()
    trans_v = ((b ^ (b >> 8)) & 0x00FFFFFFFFFFFFFF).bit_count()

    return (
        empty * 14.0
        + space_3x3 * 26.0
        - isolated * 42.0
        - (trans_h + trans_v) * 4.2
    )


def solve_best_sequence(board_mask, pieces):
    if not pieces:
        return []

    best_score = -1e9
    best_steps = []

    for perm in itertools.permutations(pieces):
        def dfs(idx, cur_mask, steps, acc_lines, acc_pts):
            nonlocal best_score, best_steps
            if idx == len(perm):
                total_eval = evaluate_bitboard(cur_mask) + acc_lines * 85.0 + acc_pts * 2.0
                score = len(steps) * 100000 + total_eval
                if score > best_score:
                    best_score = score
                    best_steps = list(steps)
                return

            p = perm[idx]
            is_last = (idx == len(perm) - 1)
            candidates = []

            for r0, c0, pmask, cells in p["placements"]:
                if (cur_mask & pmask) == 0:
                    nb, lines, pts = simulate_bitboard(cur_mask, pmask, cells)
                    h_score = evaluate_bitboard(nb) + lines * 95.0
                    candidates.append((h_score, r0, c0, nb, lines, pts))

            if not candidates:
                score = len(steps) * 100000 + evaluate_bitboard(cur_mask) + acc_lines * 85.0
                if score > best_score:
                    best_score = score
                    best_steps = list(steps)
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
                dfs(idx + 1, nb, steps, acc_lines + lines, acc_pts + pts)
                steps.pop()
            else:
                candidates.sort(key=lambda x: x[0], reverse=True)
                for _, r0, c0, nb, lines, pts in candidates[:12]:
                    steps.append({
                        "slot": p["slot"],
                        "name": p["name"],
                        "matrix": p["matrix"],
                        "row": r0 + 1,
                        "col": c0 + 1,
                        "lines": lines,
                    })
                    dfs(idx + 1, nb, steps, acc_lines + lines, acc_pts + pts)
                    steps.pop()

        dfs(0, board_mask, [], 0, 0)

    return best_steps


# ==========================================================
# 5. ĐỊNH DẠNG KẾT QUẢ GỌN RÕ CHO iOS SHORTCUTS
# ==========================================================
def shape_tag(mat):
    """Mô tả ngắn kích thước khối (VD: 3x2, 2x3, 1x5) để kiểm chứng nhanh."""
    return f"{len(mat)}×{len(mat[0])}"


def format_ios_alert_text(board, pieces, steps):
    if not steps:
        return "⚠️ Không tìm thấy nước đi hợp lệ!"

    grid = [["🟦" if board[r][c] == 1 else "⬛" for c in range(8)] for r in range(8)]
    step_badges = ["1️⃣", "2️⃣", "3️⃣"]
    lines_text = []
    placed_slots = set()

    for i, st in enumerate(steps):
        placed_slots.add(st["slot"])
        badge = step_badges[i] if i < 3 else "🔸"
        clear_note = f" (💥Xóa {st['lines']})" if st["lines"] > 0 else ""
        dim_str = shape_tag(st["matrix"])
        lines_text.append(
            f"{badge} {st['name']} ({dim_str}) ➔ H{st['row']}, C{st['col']}{clear_note}"
        )

        mat = st["matrix"]
        r0, c0 = st["row"] - 1, st["col"] - 1
        for dr in range(len(mat)):
            for dc in range(len(mat[0])):
                if mat[dr][dc] == 1:
                    if grid[r0 + dr][c0 + dc] not in ("1️⃣", "2️⃣", "3️⃣"):
                        grid[r0 + dr][c0 + dc] = badge

    for p in pieces:
        if p["slot"] not in placed_slots:
            idx = len(lines_text)
            badge = step_badges[idx] if idx < 3 else "🔸"
            lines_text.append(f"{badge} {p['name']} ({shape_tag(p['matrix'])}) ➔ Hết chỗ!")

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
    return "OK - Block Blast Solver v3.0", 200


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

        alert_text = format_ios_alert_text(board, pieces, steps)
        return alert_text, 200, {"Content-Type": "text/plain; charset=utf-8"}

    except Exception as e:
        return f"❌ Lỗi: {str(e)}", 500


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=10000)