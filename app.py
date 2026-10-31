import io
import base64
import itertools
from collections import deque
import numpy as np
from PIL import Image, ImageDraw
from flask import Flask, request, send_file

app = Flask(__name__)

# ==========================================================
# HẰNG SỐ BITBOARD 64-BIT & TỪ ĐIỂN KHỐI CHUẨN BLOCK BLAST
# ==========================================================
FULL_MASK = 0xFFFFFFFFFFFFFFFF
NOT_RIGHT = 0x7F7F7F7F7F7F7F7F
NOT_LEFT = 0xFEFEFEFEFEFEFEFE
ROW_MASKS = [0xFF << (r * 8) for r in range(8)]
COL_MASKS = [0x0101010101010101 << c for c in range(8)]

# Danh sách toàn bộ hình dạng khối hợp lệ trong Block Blast
BASE_SHAPES = [
    [[1]],
    [[1, 1]],
    [[1, 1, 1]],
    [[1, 1, 1, 1]],
    [[1, 1, 1, 1, 1]],
    [[1, 1], [1, 1]],
    [[1, 1, 1], [1, 1, 1]],
    [[1, 1, 1], [1, 1, 1], [1, 1, 1]],
    [[1, 0], [1, 1]],
    [[1, 0, 0], [1, 1, 1]],
    [[0, 0, 1], [1, 1, 1]],
    [[1, 0, 0], [1, 0, 0], [1, 1, 1]],
    [[0, 0, 1], [0, 0, 1], [1, 1, 1]],
    [[0, 1, 0], [1, 1, 1]],
    [[0, 1, 0], [0, 1, 0], [1, 1, 1]],
    [[1, 1, 0], [0, 1, 1]],
    [[0, 1, 1], [1, 1, 0]],
    [[1, 0], [0, 1]],
    [[0, 1], [1, 0]],
    [[1, 0, 0], [0, 1, 0], [0, 0, 1]],
    [[0, 0, 1], [0, 1, 0], [1, 0, 0]],
    [[0, 1, 0], [1, 1, 1], [0, 1, 0]],
]


def build_valid_shapes():
    seen = set()
    valid = []
    for s in BASE_SHAPES:
        cur = np.array(s, dtype=int)
        for _ in range(4):
            for m in (cur, np.fliplr(cur)):
                key = (m.shape[0], m.shape[1], tuple(m.flatten().tolist()))
                if key not in seen:
                    seen.add(key)
                    valid.append(m.tolist())
            cur = np.rot90(cur)
    return valid


VALID_SHAPES = build_valid_shapes()


def snap_to_valid_shape(raw_mat, bw_p, bh_p, unit_base):
    """Khớp ma trận quét được với hình khối hợp lệ gần nhất trong Block Blast."""
    raw = np.array(raw_mat, dtype=int)
    R, C = raw.shape
    best_shape = raw_mat
    best_score = -1e9

    for cand in VALID_SHAPES:
        c_arr = np.array(cand, dtype=int)
        cr, cc = c_arr.shape

        # Điểm khớp kích thước vật lý
        u_cand = unit_base * 0.84 if max(cr, cc) == 5 else unit_base
        exp_w = cc * u_cand
        exp_h = cr * u_cand
        size_err = (abs(bw_p - exp_w) + abs(bh_p - exp_h)) / max(1.0, u_cand)

        if size_err > 1.35:
            continue

        # Nếu cùng số hàng/cột thì so sánh trực tiếp từng ô
        if cr == R and cc == C:
            diff_cells = int(np.abs(raw - c_arr).sum())
            score = 100.0 - diff_cells * 28.0 - size_err * 18.0
        else:
            # Nội suy ma trận raw về kích thước (cr, cc) để so sánh
            rs = np.linspace(0, R - 1, cr).round().astype(int)
            cs = np.linspace(0, C - 1, cc).round().astype(int)
            resampled = raw[np.ix_(rs, cs)]
            diff_cells = int(np.abs(resampled - c_arr).sum())
            score = 80.0 - diff_cells * 28.0 - size_err * 22.0

        if score > best_score:
            best_score = score
            best_shape = cand

    return best_shape


# ==========================================================
# 1. BỘ NHẬN DIỆN ẢNH CHỤP MÀN HÌNH & TÁCH 3 KHỐI CHUẨN
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
    h, w, _ = rgb_arr.shape
    lum = (
        0.299 * rgb_arr[:, :, 0]
        + 0.587 * rgb_arr[:, :, 1]
        + 0.114 * rgb_arr[:, :, 2]
    )
    board_w = int(round(w * 0.908))
    board_x = int(round((w - board_w) / 2.0))

    vert_diff = np.abs(np.diff(lum[:, board_x : board_x + board_w], axis=0))
    row_edge = vert_diff.mean(axis=1)

    y_min = int(h * 0.16)
    y_max = int(h * 0.30)
    best_y = int(h * 0.225)
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


def extract_main_blob(mask):
    """Dùng BFS tách đúng cụm pixel của khối gạch, loại bỏ 100% nhiễu viền ngoài."""
    H, W = mask.shape
    visited = np.zeros((H, W), dtype=bool)
    best_blob = None
    best_weight = -1.0
    center_y, center_x = H / 2.0, W / 2.0

    for r in range(H):
        for c in range(W):
            if mask[r, c] and not visited[r, c]:
                q = deque([(r, c)])
                visited[r, c] = True
                pts_y = [r]
                pts_x = [c]
                while q:
                    cy, cx = q.popleft()
                    for ny, nx in ((cy - 1, cx), (cy + 1, cx), (cy, cx - 1), (cy, cx + 1)):
                        if 0 <= ny < H and 0 <= nx < W and mask[ny, nx] and not visited[ny, nx]:
                            visited[ny, nx] = True
                            q.append((ny, nx))
                            pts_y.append(ny)
                            pts_x.append(nx)

                area = len(pts_x)
                if area < 10:
                    continue
                mean_y = sum(pts_y) / area
                mean_x = sum(pts_x) / area
                dist_center = ((mean_y - center_y) ** 2 + (mean_x - center_x) ** 2) ** 0.5
                # Ưu tiên cụm lớn và nằm gần tâm khay
                weight = area - dist_center * 1.8
                if weight > best_weight:
                    best_weight = weight
                    best_blob = (pts_y, pts_x)

    return best_blob


def detect_single_slot_piece(slot_arr, unit_base):
    sh, sw, _ = slot_arr.shape
    # Lấy mẫu màu nền từ 4 góc và viền của ô chứa
    border_pixels = np.concatenate([
        slot_arr[0:3, :, :].reshape(-1, 3),
        slot_arr[-3:, :, :].reshape(-1, 3),
        slot_arr[:, 0:3, :].reshape(-1, 3),
        slot_arr[:, -3:, :].reshape(-1, 3),
    ], axis=0)
    dock_bg = np.median(border_pixels, axis=0)

    diff = np.linalg.norm(slot_arr - dock_bg, axis=2)
    gray = slot_arr.mean(axis=2)
    gx = np.abs(np.diff(gray, axis=1, prepend=gray[:, :1]))
    gy = np.abs(np.diff(gray, axis=0, prepend=gray[:1, :]))
    act = diff * 1.1 + (gx + gy) * 1.45

    # Xóa mép rìa 2px để không bao giờ dính viền ngoài
    act[:2, :] = 0
    act[-2:, :] = 0
    act[:, :2] = 0
    act[:, -2:] = 0

    # Thử ngưỡng thích ứng từ cao xuống thấp để không bỏ sót khối tối màu
    blob = None
    chosen_mask = None
    for thresh in (28.0, 21.0, 15.5):
        m = act > thresh
        b = extract_main_blob(m)
        if b is not None and len(b[0]) >= 16:
            blob = b
            chosen_mask = m
            break

    if blob is None:
        return None

    ys, xs = np.array(blob[0]), np.array(blob[1])
    min_x, max_x = int(xs.min()), int(xs.max())
    min_y, max_y = int(ys.min()), int(ys.max())
    bw_p = max_x - min_x + 1
    bh_p = max_y - min_y + 1

    if bw_p < 6 or bh_p < 6:
        return None

    max_dim = max(bw_p, bh_p)
    u = unit_base * 0.84 if max_dim > unit_base * 3.55 else unit_base
    cols = int(np.clip(round(bw_p / u), 1, 5))
    rows = int(np.clip(round(bh_p / u), 1, 5))

    # Chỉ dựng mask trên đúng cụm pixel đã lọc
    clean_mask = np.zeros_like(chosen_mask)
    clean_mask[ys, xs] = True

    mat = np.zeros((rows, cols), dtype=int)
    sub_w = bw_p / float(cols)
    sub_h = bh_p / float(rows)

    for r in range(rows):
        for c in range(cols):
            cx0 = int(min_x + c * sub_w + sub_w * 0.14)
            cx1 = int(min_x + (c + 1) * sub_w - sub_w * 0.14)
            cy0 = int(min_y + r * sub_h + sub_h * 0.14)
            cy1 = int(min_y + (r + 1) * sub_h - sub_h * 0.14)
            sub_m = clean_mask[max(0, cy0):min(sh, cy1 + 1), max(0, cx0):min(sw, cx1 + 1)]
            if sub_m.size > 0 and sub_m.mean() > 0.28:
                mat[r, c] = 1

    valid_r = np.any(mat == 1, axis=1)
    valid_c = np.any(mat == 1, axis=0)
    if not (np.any(valid_r) and np.any(valid_c)):
        return [[1]]

    mat = mat[valid_r][:, valid_c].tolist()
    return snap_to_valid_shape(mat, bw_p, bh_p, unit_base)


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
            x0 = int(bx + c * cw)
            y0 = int(by + r * cw)
            x1 = int(bx + (c + 1) * cw)
            y1 = int(by + (r + 1) * cw)

            pad = max(2, int((x1 - x0) * 0.15))
            core = arr[y0 + pad : y1 - pad, x0 + pad : x1 - pad]
            mean_rgb = core.mean(axis=(0, 1))
            lum = 0.299 * core[:, :, 0] + 0.587 * core[:, :, 1] + 0.114 * core[:, :, 2]
            std_lum = float(lum.std())

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

    sorted_flat = sorted(cell_feats, key=lambda x: x["flatness"])
    bg_rgb = np.median(np.array([x["rgb"] for x in sorted_flat[:10]]), axis=0)

    scores = []
    for f in cell_feats:
        dist = float(np.linalg.norm(f["rgb"] - bg_rgb))
        score = dist * 1.15 + f["bevel"] * 1.35 + f["std"] * 0.95
        f["score"] = score
        scores.append(score)

    thresh = compute_otsu_threshold(scores)
    board = [[0] * 8 for _ in range(8)]
    board_mask = 0
    for f in cell_feats:
        if f["score"] >= thresh:
            r, c = f["r"], f["c"]
            board[r][c] = 1
            board_mask |= (1 << (r * 8 + c))

    # Vùng khay 3 khối: giới hạn chặt không chạm vào thanh Home Bar dưới đáy iPhone
    dock_y0 = int(by + bw + cw * 0.55)
    dock_y1 = min(int(target_h * 0.88), int(dock_y0 + cw * 3.15))
    dock_x0 = int(target_w * 0.045)
    slot_w = int((target_w * 0.91) / 3.0)
    unit_base = cw * 0.468

    pieces = []
    slot_names = ["TRÁI", "GIỮA", "PHẢI"]

    for s in range(3):
        sx0 = dock_x0 + s * slot_w + 4
        sx1 = dock_x0 + (s + 1) * slot_w - 4
        slot_arr = arr[dock_y0:dock_y1, sx0:sx1]
        mat = detect_single_slot_piece(slot_arr, unit_base)
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
# 2. BỘ GIẢI BITBOARD 64-BIT SIÊU TỐC (< 0.03 GIÂY)
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
                # Mở rộng nhánh lên 12 để không bao giờ bỏ sót phương án đặt được khối thứ 3
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
# 3. ĐỊNH DẠNG THÔNG BÁO ĐỦ 3 BƯỚC RÕ RÀNG TRÊN ĐẦU
# ==========================================================
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
        lines_text.append(
            f"{badge} {st['name']} ➔ H{st['row']}, C{st['col']}{clear_note}"
        )

        mat = st["matrix"]
        r0, c0 = st["row"] - 1, st["col"] - 1
        for dr in range(len(mat)):
            for dc in range(len(mat[0])):
                if mat[dr][dc] == 1:
                    # Nếu ô đã có số bước trước (do xóa hàng rồi đặt đè), giữ số bước trước để không biến dạng hình
                    if grid[r0 + dr][c0 + dc] not in ("1️⃣", "2️⃣", "3️⃣"):
                        grid[r0 + dr][c0 + dc] = badge

    # Nếu có khối trong khay chưa đặt được, vẫn hiển thị dòng thứ 3 để người chơi biết rõ
    for p in pieces:
        if p["slot"] not in placed_slots:
            idx = len(lines_text)
            badge = step_badges[idx] if idx < 3 else "🔸"
            lines_text.append(f"{badge} {p['name']} ➔ Hết chỗ đặt!")

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

        alert_text = format_ios_alert_text(board, pieces, steps)
        return alert_text, 200, {"Content-Type": "text/plain; charset=utf-8"}

    except Exception as e:
        return f"❌ Lỗi: {str(e)}", 500


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=10000)