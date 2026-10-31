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
# 3. NHẬN DIỆN CHÍNH XÁC 3 KHỐI CHỜ (STEP-GRID SAMPLING)
# ==========================================================
def extract_clean_piece_blob(mask):
    """Tách cụm liên thông của khối gạch ở giữa ô khay, bỏ qua mọi vệt nhiễu chạm mép."""
    H, W = mask.shape
    visited = np.zeros((H, W), dtype=bool)
    best_blob = None
    best_score = -1e9
    cy_mid, cx_mid = H / 2.0, W / 2.0

    for r in range(H):
        for c in range(W):
            if mask[r, c] and not visited[r, c]:
                q = deque([(r, c)])
                visited[r, c] = True
                ys = [r]
                xs = [c]
                touches_top_or_bot = (r <= 1 or r >= H - 2)

                while q:
                    y, x = q.popleft()
                    if y <= 1 or y >= H - 2:
                        touches_top_or_bot = True
                    for ny, nx in ((y - 1, x), (y + 1, x), (y, x - 1), (y, x + 1)):
                        if 0 <= ny < H and 0 <= nx < W and mask[ny, nx] and not visited[ny, nx]:
                            visited[ny, nx] = True
                            q.append((ny, nx))
                            ys.append(ny)
                            xs.append(nx)

                area = len(xs)
                if area < 14:
                    continue

                # Nếu cụm chạm sát mép trên/dưới (dính viền bàn cờ hoặc banner quảng cáo) -> phạt nặng
                mean_y = sum(ys) / area
                mean_x = sum(xs) / area
                dist = ((mean_y - cy_mid) ** 2 + (mean_x - cx_mid) ** 2) ** 0.5
                penalty = 500.0 if touches_top_or_bot else 0.0
                score = area - dist * 2.2 - penalty

                if score > best_score:
                    best_score = score
                    best_blob = (np.array(ys), np.array(xs))

    return best_blob


def detect_three_dock_pieces(arr, bx, by, bw):
    """
    Định vị vùng khay dưới bàn cờ (tránh hoàn toàn viền đáy bàn cờ và banner quảng cáo dưới cùng),
    sau đó đo kích thước từng khối theo đơn vị ô chuẩn u để lấy ma trận chính xác.
    """
    h, w, _ = arr.shape
    cw = bw / 8.0

    # Khoảng cách từ đáy bàn cờ xuống khay: bắt đầu sau đáy bàn cờ tối thiểu 0.45*cw
    # và kết thúc trước vùng banner quảng cáo đáy màn hình (h * 0.86)
    search_y0 = int(round(by + bw + cw * 0.42))
    search_y1 = min(int(round(h * 0.855)), int(round(by + bw + cw * 3.85)))

    if search_y1 - search_y0 < 30:
        search_y0 = int(round(by + bw + 8))
        search_y1 = min(h - 10, search_y0 + int(round(cw * 3.0)))

    dock_band = arr[search_y0:search_y1, bx : bx + bw]
    # Màu nền chung của vùng khay (hồng nhạt hoặc xanh đậm) lấy từ median toàn dải
    dock_bg = np.median(dock_band.reshape(-1, 3), axis=0)

    # Tìm tâm dọc (center_y) nơi có mật độ khối cao nhất để cắt khung khay chuẩn xác
    band_diff = np.linalg.norm(dock_band - dock_bg, axis=2)
    row_activity = (band_diff > 20.0).mean(axis=1)
    active_rows = np.where(row_activity > 0.04)[0]

    if len(active_rows) > 4:
        ry0 = max(0, int(active_rows.min()) - int(cw * 0.35))
        ry1 = min(dock_band.shape[0], int(active_rows.max()) + int(cw * 0.35))
    else:
        ry0 = 0
        ry1 = dock_band.shape[0]

    dock_y0 = search_y0 + ry0
    dock_y1 = search_y0 + ry1

    slot_w = bw / 3.0
    # Trong Block Blast, 1 ô gạch ở khay chờ có kích thước bằng ~0.465 lần ô bàn cờ
    # (Riêng thanh 5 ô 1x5 hoặc 5x1 sẽ thu nhỏ nhẹ còn ~0.415 lần ô bàn cờ)
    unit_std = cw * 0.465

    pieces = []
    slot_names = ["TRÁI", "GIỮA", "PHẢI"]

    for s in range(3):
        sx0 = int(round(bx + s * slot_w + 4))
        sx1 = int(round(bx + (s + 1) * slot_w - 4))
        slot_arr = arr[dock_y0:dock_y1, sx0:sx1]
        sh, sw, _ = slot_arr.shape
        if sh < 10 or sw < 10:
            continue

        # Lấy màu nền cục bộ từ mép trái/phải của slot
        edge_px = np.concatenate([
            slot_arr[:, :3, :].reshape(-1, 3),
            slot_arr[:, -3:, :].reshape(-1, 3),
        ], axis=0)
        local_bg = np.median(edge_px, axis=0)

        color_dist = np.linalg.norm(slot_arr - local_bg, axis=2)
        gray = slot_arr.mean(axis=2)
        gx = np.abs(np.diff(gray, axis=1, prepend=gray[:, :1]))
        gy = np.abs(np.diff(gray, axis=0, prepend=gray[:1, :]))

        # Kết hợp độ lệch màu và cạnh vát 3D của viên gạch
        act = color_dist * 1.15 + (gx + gy) * 1.35
        act[:2, :] = 0
        act[-2:, :] = 0
        act[:, :2] = 0
        act[:, -2:] = 0

        blob = None
        for thresh in (25.0, 19.0, 14.5):
            m = act > thresh
            b = extract_clean_piece_blob(m)
            if b is not None and len(b[0]) >= 16:
                blob = b
                break

        if blob is None:
            continue

        ys, xs = blob
        min_x, max_x = int(np.percentile(xs, 1)), int(np.percentile(xs, 99))
        min_y, max_y = int(np.percentile(ys, 1)), int(np.percentile(ys, 99))
        bw_p = max_x - min_x + 1
        bh_p = max_y - min_y + 1

        if bw_p < 6 or bh_p < 6:
            continue

        # Xác định số hàng (rows) và số cột (cols) của khối dựa trên kích thước chuẩn
        ratio_max = max(bw_p, bh_p) / unit_std
        if ratio_max >= 3.65:
            # Khối có chiều dài 4 hoặc 5 ô (thu nhỏ nhẹ trong khay)
            u_eff = unit_std * 0.85
        else:
            u_eff = unit_std

        cols = int(np.clip(round(bw_p / u_eff), 1, 5))
        rows = int(np.clip(round(bh_p / u_eff), 1, 5))

        # Tạo mask sạch của riêng cụm gạch
        clean_mask = np.zeros((sh, sw), dtype=bool)
        clean_mask[ys, xs] = True

        mat = np.zeros((rows, cols), dtype=int)
        sub_w = bw_p / float(cols)
        sub_h = bh_p / float(rows)

        for r in range(rows):
            for c in range(cols):
                # Lấy vùng lõi 65% giữa mỗi ô con để kiểm tra có gạch hay khuyết
                cx0 = int(round(min_x + (c + 0.18) * sub_w))
                cx1 = int(round(min_x + (c + 0.82) * sub_w))
                cy0 = int(round(min_y + (r + 0.18) * sub_h))
                cy1 = int(round(min_y + (r + 0.82) * sub_h))

                sub_m = clean_mask[max(0, cy0) : min(sh, cy1 + 1), max(0, cx0) : min(sw, cx1 + 1)]
                sub_act = act[max(0, cy0) : min(sh, cy1 + 1), max(0, cx0) : min(sw, cx1 + 1)]

                if sub_m.size > 0 and (sub_m.mean() >= 0.32 or sub_act.mean() >= 22.0):
                    mat[r, c] = 1

        valid_r = np.any(mat == 1, axis=1)
        valid_c = np.any(mat == 1, axis=0)
        if np.any(valid_r) and np.any(valid_c):
            mat_list = mat[valid_r][:, valid_c].tolist()
            pieces.append({
                "slot": s + 1,
                "name": slot_names[s],
                "matrix": mat_list,
                "placements": precompute_placements(mat_list),
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