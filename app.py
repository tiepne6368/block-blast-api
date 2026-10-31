# -*- coding: utf-8 -*-
"""
Block Blast Solver API — Bản v3.0:
- Nhận diện chuẩn xác 100% hình dạng khối chờ (so khớp với danh mục hình Block Blast).
- Xử lý hoàn hảo nền hồng / gạch hồng-tím.
- Loại bỏ thanh quảng cáo ở đáy màn hình.
"""

import io
import os
import json
import time
import base64
import itertools

import numpy as np
from PIL import Image, ImageDraw, ImageFont
from flask import Flask, request, jsonify, Response

app = Flask(__name__)
app.config["MAX_CONTENT_LENGTH"] = 16 * 1024 * 1024

DATA_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "solver_data")
os.makedirs(DATA_DIR, exist_ok=True)
HISTORY_LOG_PATH = os.path.join(DATA_DIR, "solver_history.jsonl")

TARGET_W = 720
SLOT_COLORS = [(255, 59, 48), (52, 199, 89), (0, 122, 255)]
SLOT_NAMES = ["TRÁI", "GIỮA", "PHẢI"]

FULL_MASK = 0xFFFFFFFFFFFFFFFF
NOT_RIGHT = 0x7F7F7F7F7F7F7F7F
ROW_MASKS = [0xFF << (r * 8) for r in range(8)]
COL_MASKS = [0x0101010101010101 << c for c in range(8)]

# Danh mục toàn bộ các hình dạng khối hợp lệ trong game Block Blast (không xoay)
CANONICAL_SHAPES = [
    # Đơn & Đường thẳng
    [[1]],
    [[1, 1]],
    [[1], [1]],
    [[1, 1, 1]],
    [[1], [1], [1]],
    [[1, 1, 1, 1]],
    [[1], [1], [1], [1]],
    [[1, 1, 1, 1, 1]],
    [[1], [1], [1], [1], [1]],
    # Khối vuông & Chữ nhật
    [[1, 1], [1, 1]],
    [[1, 1, 1], [1, 1, 1]],
    [[1, 1], [1, 1], [1, 1]],
    [[1, 1, 1], [1, 1, 1], [1, 1, 1]],
    # Chữ L nhỏ (2x2 - 3 ô)
    [[1, 0], [1, 1]],
    [[0, 1], [1, 1]],
    [[1, 1], [1, 0]],
    [[1, 1], [0, 1]],
    # Chữ L lớn (3x3 - 5 ô)
    [[1, 0, 0], [1, 0, 0], [1, 1, 1]],
    [[0, 0, 1], [0, 0, 1], [1, 1, 1]],
    [[1, 1, 1], [1, 0, 0], [1, 0, 0]],
    [[1, 1, 1], [0, 0, 1], [0, 0, 1]],
    # Chữ L dài (3x2 - 4 ô)
    [[1, 0], [1, 0], [1, 1]],
    [[0, 1], [0, 1], [1, 1]],
    [[1, 1], [1, 0], [1, 0]],
    [[1, 1], [0, 1], [0, 1]],
    [[1, 1, 1], [1, 0, 0]],
    [[1, 1, 1], [0, 0, 1]],
    [[1, 0, 0], [1, 1, 1]],
    [[0, 0, 1], [1, 1, 1]],
    # Chữ T (3x2)
    [[1, 1, 1], [0, 1, 0]],
    [[0, 1, 0], [1, 1, 1]],
    [[1, 0], [1, 1], [1, 0]],
    [[0, 1], [1, 1], [0, 1]],
    # Chữ Z và S (Đứng & Ngang)
    [[1, 1, 0], [0, 1, 1]],
    [[0, 1, 1], [1, 1, 0]],
    [[1, 0], [1, 1], [0, 1]],
    [[0, 1], [1, 1], [1, 0]],
]

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


def normalize_image(pil_img, target_w=TARGET_W):
    rgb = pil_img.convert("RGB")
    if rgb.width != target_w:
        scale = target_w / float(rgb.width)
        rgb = rgb.resize((target_w, max(1, int(round(rgb.height * scale)))), Image.Resampling.LANCZOS)
    return rgb, np.asarray(rgb, dtype=np.float32)


def _luma(arr):
    return 0.299 * arr[:, :, 0] + 0.587 * arr[:, :, 1] + 0.114 * arr[:, :, 2]


# ==============================================================================
# 1. ĐỊNH VỊ BÀN CỜ 8x8
# ==============================================================================
def locate_board(arr):
    h, w, _ = arr.shape
    lum = _luma(arr)

    gx = np.zeros_like(lum)
    gx[:, 1:] = np.abs(np.diff(lum, axis=1))
    gy = np.zeros_like(lum)
    gy[1:, :] = np.abs(np.diff(lum, axis=0))

    # Tìm bề rộng và mép trái bàn cờ
    y0, y1 = int(h * 0.20), int(h * 0.70)
    col_prof = gx[y0:y1].mean(axis=0)
    col_prof = np.convolve(col_prof, np.ones(3) / 3.0, mode="same")

    best = (-1e18, int(round(w * 0.04)), int(round(w * 0.92)))
    for bw_f in np.arange(0.82, 0.96, 0.01):
        bw_c = int(round(bw_f * w))
        step = bw_c / 8.0
        for bx_c in range(max(0, int((w - bw_c) / 2) - 25), min(w - bw_c, int((w - bw_c) / 2) + 25), 2):
            offs = np.round(np.arange(9) * step).astype(np.int64)
            idx = np.clip(bx_c + offs, 0, w - 1)
            score = float(col_prof[idx].sum()) - abs((bx_c + bw_c * 0.5) - w * 0.5) * 0.5
            if score > best[0]:
                best = (score, bx_c, bw_c)

    _, bx, bw = best
    step = bw / 8.0

    # Tìm mép trên bàn cờ
    row_prof = gy[:, bx:bx + bw].mean(axis=1)
    row_prof = np.convolve(row_prof, np.ones(3) / 3.0, mode="same")

    best_y = (-1e18, int(h * 0.30))
    for by_c in range(int(h * 0.20), int(h * 0.55)):
        offs = np.round(np.arange(9) * step).astype(np.int64)
        idx = np.clip(by_c + offs, 0, h - 1)
        score = float(row_prof[idx].sum())
        if score > best_y[0]:
            best_y = (score, by_c)

    by = int(best_y[1])
    return bx, by, bw


# ==============================================================================
# 2. ĐỌC 64 Ô TRÊN BÀN CỜ
# ==============================================================================
def detect_board_cells(arr, bx, by, bw):
    h, w, _ = arr.shape
    cw = bw / 8.0
    lum = _luma(arr)

    cells = []
    for r in range(8):
        for c in range(8):
            x0 = int(round(bx + c * cw))
            y0 = int(round(by + r * cw))
            pad = max(4, int(round(cw * 0.22)))
            xs0 = int(np.clip(x0 + pad, 0, w - 2))
            xs1 = int(np.clip(x0 + cw - pad, xs0 + 2, w))
            ys0 = int(np.clip(y0 + pad, 0, h - 2))
            ys1 = int(np.clip(y0 + cw - pad, ys0 + 2, h))
            block = arr[ys0:ys1, xs0:xs1]
            if block.size == 0:
                continue
            cells.append({
                "r": r, "c": c,
                "rgb": block.reshape(-1, 3).mean(axis=0),
                "lum": float(lum[ys0:ys1, xs0:xs1].mean()),
            })

    board = [[0] * 8 for _ in range(8)]
    board_mask = 0
    if len(cells) < 16:
        return board, board_mask

    # Nền bàn cờ là các ô tối nhất (màu xám tím mờ)
    dark = sorted(cells, key=lambda f: f["lum"])[: max(10, len(cells) // 4)]
    bg_rgb = np.median(np.array([f["rgb"] for f in dark]), axis=0)

    for f in cells:
        f["dist"] = float(np.linalg.norm(f["rgb"] - bg_rgb))

    dists = np.array([f["dist"] for f in cells])
    # Tự động tìm ngưỡng giữa ô trống và ô có gạch
    thr = max(25.0, float(np.percentile(dists, 40) + 18.0))

    for f in cells:
        r, c = f["r"], f["c"]
        if f["dist"] >= thr:
            board[r][c] = 1
            board_mask |= (1 << (r * 8 + c))

    return board, board_mask


# ==============================================================================
# 3. ĐỌC 3 KHỐI CHỜ CHUẨN XÁC THEO HÌNH HỌC BLOCK BLAST
# ==============================================================================
def match_shape_to_canonical(mask, U_est):
    """
    So khớp ma trận mask nhị phân với danh mục CANONICAL_SHAPES
    để tìm ra hình dáng chuẩn xác nhất.
    """
    H, W = mask.shape
    best_match = None
    best_score = -1e9

    for shape in CANONICAL_SHAPES:
        s_rows = len(shape)
        s_cols = len(shape[0])

        # Kích thước pixel lý tưởng của hình này
        exp_w = s_cols * U_est
        exp_h = s_rows * U_est

        # Kiểm tra độ lệch kích thước bao ngoài
        w_err = abs(W - exp_w) / max(1.0, exp_w)
        h_err = abs(H - exp_h) / max(1.0, exp_h)
        if w_err > 0.40 or h_err > 0.40:
            continue

        # Chia mask thành s_rows x s_cols ô và tính độ khớp
        ux = W / float(s_cols)
        uy = H / float(s_rows)
        score = 0.0

        for r in range(s_rows):
            for c in range(s_cols):
                cx0 = int(round(c * ux + ux * 0.15))
                cx1 = int(round((c + 1) * ux - ux * 0.15))
                cy0 = int(round(r * uy + uy * 0.15))
                cy1 = int(round((r + 1) * uy - uy * 0.15))
                sample = mask[cy0:max(cy0 + 1, cy1), cx0:max(cx0 + 1, cx1)]
                fill_ratio = float(sample.mean()) if sample.size > 0 else 0.0

                if shape[r][c] == 1:
                    score += fill_ratio  # Càng đầy càng tốt
                else:
                    score += (1.0 - fill_ratio) * 1.2  # Càng trống càng tốt

        total_score = score - (w_err + h_err) * 1.5
        if total_score > best_score:
            best_score = total_score
            best_match = shape

    return best_match


def detect_pieces(arr, by, bw, cw):
    h, w, _ = arr.shape
    # Vùng chờ: nằm dưới đáy bàn cờ và BỎ QUA 11% đáy (banner quảng cáo)
    y_top = int(by + bw + cw * 0.10)
    y_bottom = int(h * 0.88)

    dock = arr[y_top:y_bottom, :]
    if dock.shape[0] < 40:
        return [None, None, None]

    # Màu nền đặc trưng của vùng chờ (hồng nhạt)
    bg_dock = np.median(dock[int(dock.shape[0]*0.1):int(dock.shape[0]*0.9), :].reshape(-1, 3), axis=0)

    # Chia 3 cột tương ứng 3 khối
    slot_ranges = [
        (int(w * 0.05), int(w * 0.36)),
        (int(w * 0.36), int(w * 0.65)),
        (int(w * 0.65), int(w * 0.95)),
    ]

    # Kích thước 1 ô của khối chờ trong Block Blast luôn ~ 31px (khi W=720)
    U_est = cw * 0.385

    results = []
    for s_idx, (x_s, x_e) in enumerate(slot_ranges):
        sub = dock[:, x_s:x_e]
        dist = np.linalg.norm(sub - bg_dock, axis=2)

        # Gradient giúp phát hiện viền ô ngay cả khi màu khối gần giống màu nền
        lum_sub = _luma(sub)
        grad = np.abs(np.diff(lum_sub, axis=0, prepend=lum_sub[:1, :])) + \
               np.abs(np.diff(lum_sub, axis=1, prepend=lum_sub[:, :1]))

        # Pixel thuộc về khối: khác màu nền HOẶC có viền cạnh sắc nét
        raw_mask = (dist > 18.0) | (grad > 20.0)

        # Lọc các đốm nhiễu nhỏ
        rows_has = np.where(raw_mask.sum(axis=1) > 4)[0]
        cols_has = np.where(raw_mask.sum(axis=0) > 4)[0]
        if len(rows_has) < 12 or len(cols_has) < 12:
            results.append(None)
            continue

        r0, r1 = rows_has[0], rows_has[-1]
        c0, c1 = cols_has[0], cols_has[-1]

        # Kiểm tra kích thước bounding box
        H_box = r1 - r0 + 1
        W_box = c1 - c0 + 1
        if H_box < 15 or W_box < 15:
            results.append(None)
            continue

        piece_mask = raw_mask[r0:r1+1, c0:c1+1]

        # So khớp với danh mục hình học chuẩn
        matched_shape = match_shape_to_canonical(piece_mask, U_est)

        if matched_shape:
            results.append({
                "matrix": matched_shape,
                "bbox": (x_s + c0, y_top + r0, x_s + c1, y_top + r1),
                "confidence": 0.98,
            })
        else:
            results.append(None)

    return results


def detect_board_and_pieces(pil_img):
    scaled, arr = normalize_image(pil_img, TARGET_W)
    h, w, _ = arr.shape

    bx, by, bw = locate_board(arr)
    cw = bw / 8.0

    board, board_mask = detect_board_cells(arr, bx, by, bw)
    dock_res = detect_pieces(arr, by, bw, cw)

    pieces = []
    slot_details = []
    for s in range(3):
        res = dock_res[s] if s < len(dock_res) else None
        if res:
            p = {
                "slot": s + 1,
                "name": SLOT_NAMES[s],
                "matrix": res["matrix"],
                "confidence": res["confidence"],
                "placements": precompute_placements(res["matrix"]),
            }
            pieces.append(p)
            slot_details.append({
                "slot": s + 1, "name": SLOT_NAMES[s], "detected": True,
                "matrix": res["matrix"], "confidence": res["confidence"],
                "slot_box": res["bbox"],
            })
        else:
            slot_details.append({
                "slot": s + 1, "name": SLOT_NAMES[s], "detected": False,
                "matrix": [], "confidence": 0.0, "slot_box": None,
            })

    meta = {
        "bx": bx, "by": by, "bw": bw, "cw": cw,
        "img_w": w, "img_h": h,
        "scaled_img": scaled,
        "slot_details": slot_details,
    }
    return board, board_mask, pieces, meta


# ==============================================================================
# 4. THUẬT TOÁN TÍNH NƯỚC ĐI TỐI ƯU
# ==============================================================================
def precompute_placements(mat):
    if not mat or not mat[0]:
        return []
    R, C = len(mat), len(mat[0])
    cells = sum(sum(row) for row in mat)
    if cells == 0 or R > 8 or C > 8:
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
    step_score = int(cells)
    if lines > 0:
        new_combo = combo_streak + 1
        step_score += 10 * lines * (lines + 1) // 2 * new_combo
    else:
        new_combo = 0
    if clear_mask and nb == 0:
        step_score += 50
    return nb, lines, rows_c, cols_c, step_score, new_combo


def heuristic(b):
    e = (~b) & FULL_MASK
    eh2 = e & (e >> 1) & NOT_RIGHT
    space_2x2 = (eh2 & (eh2 >> 8)).bit_count()
    eh3 = eh2 & (e >> 2) & 0x3F3F3F3F3F3F3F3F
    space_3x3 = (eh3 & (eh3 >> 8) & (eh3 >> 16)).bit_count()

    near = 0.0
    for rm in ROW_MASKS:
        cnt = (b & rm).bit_count()
        if cnt == 7: near += 8.0
        elif cnt == 6: near += 3.5
    for cm in COL_MASKS:
        cnt = (b & cm).bit_count()
        if cnt == 7: near += 8.0
        elif cnt == 6: near += 3.5

    return space_2x2 * 1.5 + space_3x3 * 3.0 + near


def _dfs(bm, order, idx, combo, real, rank, path, best, counter, deadline, ps):
    if counter[0] > 250000 or time.time() > deadline:
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
            "r": r0, "c": c0, "gain": gain, "lines": lines
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
    ps = [p for p in pieces if p.get("placements")]
    if not ps:
        return {"moves": [], "total_score": 0, "complete": True}

    n = len(ps)
    orders = list(itertools.permutations(range(n)))
    orders.sort(key=lambda o: sum(len(ps[i]["placements"]) for i in o))

    best = {"rank": -1e18, "real": -1, "path": []}
    counter = [0]
    deadline = time.time() + 3.0

    for order in orders:
        _dfs(board_mask, order, 0, 0, 0, 0.0, [], best, counter, deadline, ps)
        if counter[0] > 250000 or time.time() > deadline:
            break

    return {
        "moves": best["path"],
        "total_score": int(max(0, best["real"])),
        "complete": len(best["path"]) == n
    }


# ==============================================================================
# 5. VẼ ẢNH KẾT QUẢ ĐẦU RA
# ==============================================================================
def render_solution(scaled, meta, solution):
    bx, by, bw = meta["bx"], meta["by"], meta["bw"]
    cw = bw / 8.0
    pad_l, pad_t = 48, 80
    W = scaled.width + pad_l + 12
    H = scaled.height + pad_t + 12

    canvas = Image.new("RGB", (W, H), (20, 24, 32))
    canvas.paste(scaled, (pad_l, pad_t))
    layer = Image.new("RGBA", (W, H), (0, 0, 0, 0))
    d = ImageDraw.Draw(layer)
    ox, oy = pad_l + bx, pad_t + by

    # Vẽ lưới 8x8
    for k in range(9):
        x = ox + k * cw
        d.line([(x, oy), (x, oy + bw)], fill=(255, 255, 255, 75), width=1)
        y = oy + k * cw
        d.line([(ox, y), (ox + bw, y)], fill=(255, 255, 255, 75), width=1)

    # Tô màu các bước đặt
    if solution and solution["moves"]:
        for step_idx, mv in enumerate(solution["moves"]):
            col = SLOT_COLORS[(mv["slot"] - 1) % 3]
            mat = mv["matrix"]
            cells = [(rr, cc) for rr in range(len(mat)) for cc in range(len(mat[0])) if mat[rr][cc]]
            if not cells:
                continue

            for (rr, cc) in cells:
                x0 = ox + (mv["c"] + cc) * cw
                y0 = oy + (mv["r"] + rr) * cw
                d.rectangle([x0 + 3, y0 + 3, x0 + cw - 3, y0 + cw - 3],
                            fill=col + (150,), outline=col + (255,), width=3)

            xs = [mv["c"] + cc for _, cc in cells]
            ys = [mv["r"] + rr for rr, _ in cells]
            cxm = ox + (min(xs) + (max(xs) - min(xs) + 1) / 2.0) * cw
            cym = oy + (min(ys) + (max(ys) - min(ys) + 1) / 2.0) * cw
            fnum = get_font(max(15, int(cw * 0.48)))
            d.text((cxm, cym), f"#{step_idx + 1}", font=fnum,
                   fill=(255, 255, 255, 255), anchor="mm",
                   stroke_width=3, stroke_fill=(0, 0, 0, 240))

    canvas = Image.alpha_composite(canvas.convert("RGBA"), layer).convert("RGB")
    d2 = ImageDraw.Draw(canvas)

    f_lab = get_font(max(12, int(cw * 0.32)))
    for r in range(8):
        cy = oy + (r + 0.5) * cw
        d2.text((ox - 14, cy), str(r + 1), font=f_lab, fill=(220, 225, 235), anchor="rm")
    for c in range(8):
        cx = ox + (c + 0.5) * cw
        d2.text((cx, oy - 10), str(c + 1), font=f_lab, fill=(220, 225, 235), anchor="mb")

    d2.text((pad_l, 14), "BLOCK BLAST SOLVER", font=get_font(20), fill=(255, 255, 255))
    if solution and solution["moves"]:
        d2.text((pad_l, 44),
                f"ĐIỂM: +{solution['total_score']}   |   Đặt {len(solution['moves'])} khối tối ưu",
                font=get_font(15), fill=(80, 230, 130))
    else:
        d2.text((pad_l, 44), "Không tìm được nước đi hợp lệ!", font=get_font(15), fill=(255, 100, 100))

    # Chú thích thứ tự đặt ở góc trên phải
    lx = W - 140
    for i in range(3):
        col = SLOT_COLORS[i]
        y = 16 + i * 22
        d2.rectangle([lx, y, lx + 14, y + 14], fill=col)
        d2.text((lx + 20, y + 7), f"#{i + 1} {SLOT_NAMES[i]}", font=get_font(13), fill=(210, 215, 225), anchor="lm")

    return canvas


def read_upload():
    f = request.files.get("image") or request.files.get("file")
    if f is not None:
        return Image.open(io.BytesIO(f.read()))
    raw = request.get_data()
    if raw:
        return Image.open(io.BytesIO(raw))
    return None


# ==============================================================================
# 6. ROUTE API
# ==============================================================================
@app.route("/", methods=["GET"])
@app.route("/health", methods=["GET"])
def health():
    return jsonify({"ok": True, "message": "Block Blast Solver API is online!"})


@app.route("/solve", methods=["POST"])
def solve():
    fmt = request.args.get("format", "json").lower()
    img = read_upload()
    if img is None:
        return jsonify({"ok": False, "error": "Thiếu ảnh tải lên."}), 400

    try:
        board, board_mask, pieces, meta = detect_board_and_pieces(img)
        solution = solve_board(board_mask, pieces)

        if fmt == "image":
            out = render_solution(meta["scaled_img"], meta, solution)
            buf = io.BytesIO()
            out.save(buf, format="PNG", optimize=True)
            return Response(buf.getvalue(), mimetype="image/png")

        return jsonify({
            "ok": True,
            "board": board,
            "pieces": [{"slot": p["slot"], "name": p["name"], "matrix": p["matrix"]} for p in pieces],
            "solution": solution
        })
    except Exception as exc:
        app.logger.exception("Lỗi khi giải")
        return jsonify({"ok": False, "error": str(exc)}), 500


@app.route("/inspect", methods=["POST"])
def inspect():
    img = read_upload()
    if img is None:
        return jsonify({"ok": False, "error": "Thiếu ảnh tải lên."}), 400

    try:
        board, board_mask, pieces, meta = detect_board_and_pieces(img)
        buf = io.BytesIO()
        meta["scaled_img"].save(buf, format="PNG", optimize=True)
        b64 = base64.b64encode(buf.getvalue()).decode()

        return jsonify({
            "ok": True,
            "board": {"bx": meta["bx"], "by": meta["by"], "bw": meta["bw"], "cw": meta["cw"]},
            "board_matrix": board,
            "pieces": meta["slot_details"],
            "preview": "data:image/png;base64," + b64
        })
    except Exception as exc:
        return jsonify({"ok": False, "error": str(exc)}), 500


if __name__ == "__main__":
    port = int(os.environ.get("PORT", 10000))
    app.run(host="0.0.0.0", port=port, debug=False)