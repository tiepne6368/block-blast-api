import io
import base64
import itertools
import numpy as np
from PIL import Image, ImageDraw
from flask import Flask, request, jsonify, send_file

app = Flask(__name__)


# ==========================================================
# 1. BỘ NHẬN DIỆN ẢNH CHỤP MÀN HÌNH (THEME-AGNOSTIC VISION)
# ==========================================================
def compute_otsu_threshold(values, min_t=22.0, max_t=62.0):
    arr = np.asarray(values, dtype=np.float32)
    vmin, vmax = float(arr.min()), float(arr.max())
    if vmax - vmin < 12.0:
        return 35.0
    best_t = (vmin + vmax) / 2.0
    best_var = -1.0
    for step in range(1, 32):
        t = vmin + (vmax - vmin) * (step / 32.0)
        w0 = arr[arr <= t]
        w1 = arr[arr > t]
        if len(w0) == 0 or len(w1) == 0:
            continue
        var_between = len(w0) * len(w1) * ((w0.mean() - w1.mean()) ** 2)
        if var_between > best_var:
            best_var = var_between
            best_t = t
    return float(np.clip(best_t, min_t, max_t))


def auto_locate_board(rgb_arr):
    """Tự động tìm tọa độ khung bàn cờ vuông 8x8 trên mọi tỷ lệ màn hình iPhone."""
    h, w, _ = rgb_arr.shape
    lum = (
        0.299 * rgb_arr[:, :, 0]
        + 0.587 * rgb_arr[:, :, 1]
        + 0.114 * rgb_arr[:, :, 2]
    )
    board_w = int(round(w * 0.908))
    board_x = int(round((w - board_w) / 2.0))

    # Quét cạnh ngang ở vùng 17% -> 28% chiều cao ảnh để bắt đỉnh bàn cờ
    vert_diff = np.abs(np.diff(lum[:, board_x : board_x + board_w], axis=0))
    row_edge = vert_diff.mean(axis=1)

    y_min = int(h * 0.17)
    y_max = int(h * 0.28)
    best_y = int(h * 0.228)
    best_score = -1.0

    for y in range(y_min, min(y_max, h - board_w - 10)):
        top_edge = row_edge[max(0, y - 2) : y + 3].max()
        bot_y = y + board_w
        bot_edge = row_edge[max(0, bot_y - 2) : min(len(row_edge), bot_y + 3)].max()
        score = top_edge + bot_edge * 0.85
        if score > best_score:
            best_score = score
            best_y = y

    return board_x, best_y, board_w


def detect_board_and_pieces(pil_img):
    # Chuẩn hóa chiều rộng về 480px để xử lý siêu tốc (< 0.15s)
    target_w = 480
    scale = target_w / float(pil_img.width)
    target_h = int(round(pil_img.height * scale))
    img = pil_img.resize((target_w, target_h), Image.Resampling.BILINEAR)
    arr = np.asarray(img, dtype=np.float32)

    bx, by, bw = auto_locate_board(arr)
    cw = bw / 8.0

    # --- A. NHẬN DIỆN 64 Ô TRÊN BÀN CỜ 8x8 ---
    cell_feats = []
    for r in range(8):
        for c in range(8):
            x0 = int(bx + c * cw)
            y0 = int(by + r * cw)
            x1 = int(bx + (c + 1) * cw)
            y1 = int(by + (r + 1) * cw)

            pad = max(2, int((x1 - x0) * 0.15))
            core = arr[y0 + pad : y1 - pad, x0 + pad : x1 - pad]
            mean_rgb = core.mean(axis=(0, 1))
            lum = 0.299 * core[:, :, 0] + 0.587 * core[:, :, 1] + 0.114 * core[:, :, 2]
            std_lum = float(lum.std())

            # Đo hiệu ứng vát 3D (Bevel): Khối gạch luôn có viền trên/trái sáng khác biệt viền dưới/phải
            b_strip = max(2, int(lum.shape[0] * 0.24))
            top_l = lum[:b_strip, :].mean()
            bot_l = lum[-b_strip:, :].mean()
            left_l = lum[:, :b_strip].mean()
            right_l = lum[:, -b_strip:].mean()
            bevel = abs(top_l - bot_l) + abs(left_l - right_l)

            cell_feats.append({
                "r": r,
                "c": c,
                "rgb": mean_rgb,
                "std": std_lum,
                "bevel": float(bevel),
                "flatness": std_lum + float(bevel) * 0.9,
            })

    # Lấy 10 ô phẳng nhất làm màu nền chuẩn của bàn cờ (hỗ trợ mọi màu theme)
    sorted_flat = sorted(cell_feats, key=lambda x: x["flatness"])
    bg_rgb = np.median(np.array([x["rgb"] for x in sorted_flat[:10]]), axis=0)

    scores = []
    for f in cell_feats:
        dist = float(np.linalg.norm(f["rgb"] - bg_rgb))
        score = dist * 1.15 + f["bevel"] * 1.35 + f["std"] * 0.95
        f["score"] = score
        scores.append(score)

    thresh = compute_otsu_threshold(scores)
    board = np.zeros((8, 8), dtype=int)
    for f in cell_feats:
        board[f["r"], f["c"]] = 1 if f["score"] >= thresh else 0

    # --- B. NHẬN DIỆN 3 KHỐI CHỜ PHÍA DƯỚI ---
    dock_y0 = min(target_h - 40, int(by + bw + target_h * 0.042))
    dock_y1 = min(target_h - 10, int(dock_y0 + target_h * 0.195))
    dock_x0 = int(target_w * 0.04)
    slot_w = int((target_w * 0.92) / 3.0)
    unit_base = cw * 0.472  # Kích thước 1 ô nhỏ trong khay chờ

    pieces = []
    slot_names = ["TRÁI", "GIỮA", "PHẢI"]

    for s in range(3):
        sx0 = dock_x0 + s * slot_w + 4
        sx1 = dock_x0 + (s + 1) * slot_w - 4
        slot_arr = arr[dock_y0:dock_y1, sx0:sx1]
        sh, sw, _ = slot_arr.shape

        # Lấy màu nền vùng khay từ các viền mép
        border_pixels = np.concatenate([
            slot_arr[0:3, :, :].reshape(-1, 3),
            slot_arr[-3:, :, :].reshape(-1, 3),
            slot_arr[:, 0:3, :].reshape(-1, 3),
            slot_arr[:, -3:, :].reshape(-1, 3),
        ], axis=0)
        dock_bg = np.median(border_pixels, axis=0)

        diff = np.linalg.norm(slot_arr - dock_bg, axis=2)
        # Bổ sung đạo hàm cạnh để bắt khối có màu trùng tông màu nền
        gx = np.abs(np.diff(slot_arr.mean(axis=2), axis=1, prepend=slot_arr[:, :1, :].mean(axis=2)))
        gy = np.abs(np.diff(slot_arr.mean(axis=2), axis=0, prepend=slot_arr[:1, :, :].mean(axis=2)))
        act = diff + (gx + gy) * 1.35

        mask = act > 30.0
        # Khử nhiễu điểm lẻ
        ys, xs = np.where(mask)
        if len(xs) < 22:
            continue

        min_x, max_x = int(np.percentile(xs, 2)), int(np.percentile(xs, 98))
        min_y, max_y = int(np.percentile(ys, 2)), int(np.percentile(ys, 98))
        bw_p = max_x - min_x + 1
        bh_p = max_y - min_y + 1

        if bw_p < 8 or bh_p < 8:
            continue

        max_dim = max(bw_p, bh_p)
        u = unit_base * 0.84 if max_dim > unit_base * 3.55 else unit_base
        cols = int(np.clip(round(bw_p / u), 1, 5))
        rows = int(np.clip(round(bh_p / u), 1, 5))

        mat = np.zeros((rows, cols), dtype=int)
        sub_w = bw_p / float(cols)
        sub_h = bh_p / float(rows)

        for r in range(rows):
            for c in range(cols):
                cx0 = int(min_x + c * sub_w + sub_w * 0.16)
                cx1 = int(min_x + (c + 1) * sub_w - sub_w * 0.16)
                cy0 = int(min_y + r * sub_h + sub_h * 0.16)
                cy1 = int(min_y + (r + 1) * sub_h - sub_h * 0.16)
                sub_mask = mask[max(0, cy0):min(sh, cy1), max(0, cx0):min(sw, cx1)]
                if sub_mask.size > 0 and sub_mask.mean() > 0.34:
                    mat[r, c] = 1

        # Cắt bỏ hàng/cột rỗng ở rìa nếu có
        valid_r = np.any(mat == 1, axis=1)
        valid_c = np.any(mat == 1, axis=0)
        if np.any(valid_r) and np.any(valid_c):
            mat = mat[valid_r][:, valid_c]
            pieces.append({
                "slot": s + 1,
                "name": slot_names[s],
                "matrix": mat.tolist()
            })

    meta = {"bx": bx, "by": by, "bw": bw, "cw": cw, "scaled_img": img}
    return board.tolist(), pieces, meta


# ==========================================================
# 2. THUẬT TOÁN GIẢI DFS BACKTRACKING + HEURISTIC TỐI ƯU
# ==========================================================
def can_place(board, mat, r0, c0):
    R, C = len(mat), len(mat[0])
    if r0 + R > 8 or c0 + C > 8:
        return False
    for r in range(R):
        for c in range(C):
            if mat[r][c] == 1 and board[r0 + r][c0 + c] != 0:
                return False
    return True


def simulate_move(board, mat, r0, c0):
    nb = [row[:] for row in board]
    R, C = len(mat), len(mat[0])
    cells_placed = 0
    for r in range(R):
        for c in range(C):
            if mat[r][c] == 1:
                nb[r0 + r][c0 + c] = 1
                cells_placed += 1

    # Kiểm tra xóa đồng thời cả hàng và cột
    full_rows = [r for r in range(8) if all(nb[r][c] == 1 for c in range(8))]
    full_cols = [c for c in range(8) if all(nb[r][c] == 1 for r in range(8))]

    for r in full_rows:
        for c in range(8):
            nb[r][c] = 0
    for c in full_cols:
        for r in range(8):
            nb[r][c] = 0

    lines = len(full_rows) + len(full_cols)
    pts = cells_placed + (lines * 18 + (lines - 1) * 12 if lines > 0 else 0)
    return nb, lines, pts, full_rows, full_cols


def evaluate_board(board):
    empty = 0
    isolated_holes = 0
    transitions = 0

    for r in range(8):
        for c in range(8):
            v = board[r][c]
            if v == 0:
                empty += 1
                if (
                    (r == 0 or board[r - 1][c] == 1)
                    and (r == 7 or board[r + 1][c] == 1)
                    and (c == 0 or board[r][c - 1] == 1)
                    and (c == 7 or board[r][c + 1] == 1)
                ):
                    isolated_holes += 1
            if c < 7 and v != board[r][c + 1]:
                transitions += 1
            if r < 7 and v != board[r + 1][c]:
                transitions += 1

    # Đếm số vùng trống 3x3 để đảm bảo sống lâu khi gặp khối vuông to
    space_3x3 = 0
    for r in range(6):
        for c in range(6):
            if all(board[r + dr][c + dc] == 0 for dr in range(3) for dc in range(3)):
                space_3x3 += 1

    return (
        empty * 14.0
        + space_3x3 * 26.0
        - isolated_holes * 42.0
        - transitions * 4.2
    )


def solve_best_sequence(board, pieces):
    if not pieces:
        return []

    best_score = -1e9
    best_steps = []

    for perm in itertools.permutations(pieces):
        def dfs(idx, cur_board, steps, acc_lines, acc_pts):
            nonlocal best_score, best_steps
            if idx == len(perm):
                total_eval = evaluate_board(cur_board) + acc_lines * 85.0 + acc_pts * 2.0
                if len(steps) * 10000 + total_eval > best_score:
                    best_score = len(steps) * 10000 + total_eval
                    best_steps = list(steps)
                return

            p = perm[idx]
            mat = p["matrix"]
            R, C = len(mat), len(mat[0])
            candidates = []

            for r in range(8 - R + 1):
                for c in range(8 - C + 1):
                    if can_place(cur_board, mat, r, c):
                        nb, lines, pts, f_rows, f_cols = simulate_move(cur_board, mat, r, c)
                        h_score = evaluate_board(nb) + lines * 90.0
                        candidates.append((h_score, r, c, nb, lines, pts, f_rows, f_cols))

            if not candidates:
                # Nếu bị tắc giữa chừng, vẫn lưu phương án đặt được nhiều khối nhất
                total_eval = evaluate_board(cur_board) + acc_lines * 85.0
                if len(steps) * 10000 + total_eval > best_score:
                    best_score = len(steps) * 10000 + total_eval
                    best_steps = list(steps)
                return

            # Beam search: lấy Top 7 vị trí tốt nhất mỗi bước để phản hồi dưới 0.1 giây
            candidates.sort(key=lambda x: x[0], reverse=True)
            for cand in candidates[:7]:
                _, r, c, nb, lines, pts, f_rows, f_cols = cand
                steps.append({
                    "slot": p["slot"],
                    "name": p["name"],
                    "matrix": mat,
                    "row": r + 1,
                    "col": c + 1,
                    "lines": lines,
                    "full_rows": [x + 1 for x in f_rows],
                    "full_cols": [x + 1 for x in f_cols],
                })
                dfs(idx + 1, nb, steps, acc_lines + lines, acc_pts + pts)
                steps.pop()

        dfs(0, board, [], 0, 0)

    return best_steps


# ==========================================================
# 3. TẠO BẢN ĐỒ EMOJI 8x8 & ẢNH MINH HỌA CHO iOS SHORTCUTS
# ==========================================================
def format_ios_alert_text(board, steps):
    if not steps:
        return "⚠️ Không tìm thấy nước đi hợp lệ hoặc ảnh chụp chưa rõ bàn cờ!"

    grid = [["🟦" if board[r][c] == 1 else "⬛" for c in range(8)] for r in range(8)]
    step_badges = ["1️⃣", "2️⃣", "3️⃣"]
    lines_text = []

    for i, st in enumerate(steps):
        badge = step_badges[i] if i < 3 else "🔸"
        clear_note = ""
        if st["lines"] > 0:
            clear_note = f" (💥Xóa {st['lines']} hàng/cột)"
        lines_text.append(
            f"{badge} Khối {st['name']} ➔ Hàng {st['row']}, Cột {st['col']}{clear_note}"
        )

        # Đánh dấu vị trí đặt lên bản đồ 8x8
        mat = st["matrix"]
        r0, c0 = st["row"] - 1, st["col"] - 1
        for dr in range(len(mat)):
            for dc in range(len(mat[0])):
                if mat[dr][dc] == 1:
                    grid[r0 + dr][c0 + dc] = badge

    map_str = "\n".join("".join(row) for row in grid)
    return (
        "⚡ LỜI GIẢI BLOCK BLAST:\n\n"
        + "\n".join(lines_text)
        + "\n\n📋 BẢN ĐỒ VỊ TRÍ (8×8):\n"
        + map_str
    )


def render_solution_image(meta, steps):
    img = meta["scaled_img"].copy()
    draw = ImageDraw.Draw(img, "RGBA")
    bx, by, cw = meta["bx"], meta["by"], meta["cw"]
    colors = [
        (56, 189, 248, 190),   # Bước 1: Xanh Cyan
        (250, 204, 21, 190),   # Bước 2: Vàng
        (192, 132, 252, 190),  # Bước 3: Tím
    ]

    for i, st in enumerate(steps):
        color = colors[i % len(colors)]
        mat = st["matrix"]
        r0, c0 = st["row"] - 1, st["col"] - 1
        for dr in range(len(mat)):
            for dc in range(len(mat[0])):
                if mat[dr][dc] == 1:
                    x0 = int(bx + (c0 + dc) * cw + 3)
                    y0 = int(by + (r0 + dr) * cw + 3)
                    x1 = int(bx + (c0 + dc + 1) * cw - 3)
                    y1 = int(by + (r0 + dr + 1) * cw - 3)
                    draw.rectangle([x0, y0, x1, y1], fill=color, outline=(255, 255, 255, 255), width=2)
                    draw.text((x0 + 6, y0 + 4), f"#{i+1}", fill=(0, 0, 0, 255))

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
    return "✅ Block Blast iOS Shortcut Server đang hoạt động! Gửi ảnh vào POST /solve"


@app.route("/solve", methods=["POST"])
def solve_endpoint():
    try:
        pil_img = extract_image_from_request(request)
        if pil_img is None:
            return "Không nhận được ảnh từ Phím tắt!", 400

        board, pieces, meta = detect_board_and_pieces(pil_img)
        steps = solve_best_sequence(board, pieces)

        # Nếu gọi /solve?format=image thì trả về ảnh PNG vẽ đè nước đi
        if request.args.get("format") == "image":
            buf = render_solution_image(meta, steps)
            return send_file(buf, mimetype="image/png")

        # Mặc định trả về Text kèm Bản đồ Emoji 8x8 để hiện ngay trong Thông báo iOS
        alert_text = format_ios_alert_text(board, steps)
        return alert_text, 200, {"Content-Type": "text/plain; charset=utf-8"}

    except Exception as e:
        return f"❌ Lỗi xử lý ảnh: {str(e)}", 500


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=10000)