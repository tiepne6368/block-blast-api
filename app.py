# -*- coding: utf-8 -*-
"""
Block Blast Solver API — Bản chuẩn sửa lỗi nhận diện khối chờ và bỏ qua quảng cáo đáy.
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
from flask import Flask, request, jsonify, Response

app = Flask(__name__)
app.config["MAX_CONTENT_LENGTH"] = 16 * 1024 * 1024

DATA_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "solver_data")
os.makedirs(DATA_DIR, exist_ok=True)
HISTORY_LOG_PATH = os.path.join(DATA_DIR, "solver_history.jsonl")

TARGET_W = 720
SLOT_COLORS = [(255, 60, 60), (46, 213, 115), (30, 144, 255)]
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

    # Bước 1: Quét mép trái & bề rộng bàn cờ
    y0, y1 = int(h * 0.15), int(h * 0.75)
    col_prof = gx[y0:y1].mean(axis=0)
    col_prof = np.convolve(col_prof, np.ones(3)/3.0, mode='same')

    best = (-1e18, int(round(w * 0.04)), int(round(w * 0.92)))
    for bw_f in np.arange(0.80, 0.98, 0.01):
        bw_c = int(round(bw_f * w))
        step = bw_c / 8.0
        for bx_c in range(max(0, int((w - bw_c)/2) - 30), min(w - bw_c, int((w - bw_c)/2) + 30), 2):
            offs = np.round(np.arange(9) * step).astype(np.int64)
            idx = np.clip(bx_c + offs, 0, w - 1)
            score = float(col_prof[idx].sum()) - abs((bx_c + bw_c*0.5) - w*0.5) * 0.5
            if score > best[0]:
                best = (score, bx_c, bw_c)

    _, bx, bw = best
    step = bw / 8.0

    # Bước 2: Quét mép trên (by)
    row_prof = gy[:, bx:bx + bw].mean(axis=1)
    row_prof = np.convolve(row_prof, np.ones(3)/3.0, mode='same')

    best_y = (-1e18, int(h * 0.30))
    for by_c in range(int(h * 0.15), int(h * 0.60)):
        offs = np.round(np.arange(9) * step).astype(np.int64)
        idx = np.clip(by_c + offs, 0, h - 1)
        score = float(row_prof[idx].sum())
        if score > best_y[0]:
            best_y = (score, by_c)

    by = int(best_y[1])
    return bx, by, bw


# ==============================================================================
# 2. ĐỌC 64 Ô BÀN CỜ
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

    # Ô trống là ô tối nhất (nền bàn cờ)
    dark = sorted(cells, key=lambda f: f["lum"])[: max(8, len(cells) // 4)]
    bg_rgb = np.median(np.array([f["rgb"] for f in dark]), axis=0)

    for f in cells:
        dist = float(np.linalg.norm(f["rgb"] - bg_rgb))
        f["dist"] = dist

    dists = np.array([f["dist"] for f in cells])
    # Ngưỡng phân biệt ô trống và ô có gạch
    thr = max(24.0, float(np.percentile(dists, 50) + 15.0))
    if dists.max() - dists.min() > 30.0:
        thr = min(thr, (dists.max() + dists.min()) * 0.45)

    for f in cells:
        r, c = f["r"], f["c"]
        if f["dist"] >= thr:
            board[r][c] = 1
            board_mask |= (1 << (r * 8 + c))

    return board, board_mask


# ==============================================================================
# 3. ĐỌC 3 KHỐI CHỜ (Cắt bỏ quảng cáo đáy + Xác định chuẩn kích thước ô)
# ==============================================================================
def detect_pieces(arr, by, bw, cw):
    h, w, _ = arr.shape
    # Giới hạn vùng chứa khối chờ: ngay dưới đáy bàn cờ và BỎ QUA 11% ĐÁY (banner quảng cáo)
    y_top = int(by + bw + cw * 0.12)
    y_bottom = int(h * 0.89)

    dock = arr[y_top:y_bottom, :]
    if dock.shape[0] < 40:
        return [None, None, None]

    # Nền của khu vực chờ (màu hồng của game)
    bg_dock = np.median(dock.reshape(-1, 3), axis=0)

    # Chia chiều ngang thành 3 slot cho 3 khối: TRÁI, GIỮA, PHẢI
    slot_ranges = [
        (int(w * 0.04), int(w * 0.36)),
        (int(w * 0.36), int(w * 0.65)),
        (int(w * 0.65), int(w * 0.96)),
    ]

    slot_bboxes = []
    slot_masks = []

    for s_idx, (x_s, x_e) in enumerate(slot_ranges):
        sub = dock[:, x_s:x_e]
        dist = np.linalg.norm(sub - bg_dock, axis=2)
        lum = _luma(sub)

        # Pixel thuộc khối: khác màu nền rõ rệt VÀ không phải bóng đổ tối màu
        mask = (dist > 22.0) & (lum > 45.0)

        # Lọc nhiễu
        if mask.sum() < 60:
            slot_bboxes.append(None)
            slot_masks.append(None)
            continue

        # Tìm bounding box của khối trong slot
        rows_has = np.where(mask.any(axis=1))[0]
        cols_has = np.where(mask.any(axis=0))[0]
        if len(rows_has) < 10 or len(cols_has) < 10:
            slot_bboxes.append(None)
            slot_masks.append(None)
            continue

        r0, r1 = rows_has[0], rows_has[-1]
        c0, c1 = cols_has[0], cols_has[-1]

        # Kiểm tra lại tỉ lệ
        if (r1 - r0) < 12 or (c1 - c0) < 12:
            slot_bboxes.append(None)
            slot_masks.append(None)
            continue

        slot_bboxes.append((x_s + c0, y_top + r0, x_s + c1, y_top + r1))
        slot_masks.append(mask[r0:r1+1, c0:c1+1])

    # Kích thước 1 ô của khối chờ trong Block Blast luôn khoảng ~0.38 * cw (khoảng 28px - 36px khi W=720)
    expected_unit = cw * 0.38
    unit_candidates = []

    for bbox in slot_bboxes:
        if bbox is None:
            continue
        bw_i = bbox[2] - bbox[0] + 1
        bh_i = bbox[3] - bbox[1] + 1
        # Tìm ước số gần expected_unit nhất
        for dim in (bw_i, bh_i):
            for k in range(1, 6):
                u = dim / float(k)
                if 24 <= u <= 42:
                    unit_candidates.append(u)

    U = float(np.median(unit_candidates)) if unit_candidates else expected_unit

    results = []
    for s_idx in range(3):
        bbox = slot_bboxes[s_idx]
        mask = slot_masks[s_idx]
        if bbox is None or mask is None:
            results.append(None)
            continue

        bw_i = bbox[2] - bbox[0] + 1
        bh_i = bbox[3] - bbox[1] + 1

        cols = int(np.clip(round(bw_i / U), 1, 5))
        rows = int(np.clip(round(bh_i / U), 1, 5))

        ux = bw_i / float(cols)
        uy = bh_i / float(rows)

        mat = np.zeros((rows, cols), dtype=int)
        for r in range(rows):
            for c in range(cols):
                cx0 = int(round(c * ux + ux * 0.20))
                cx1 = int(round((c + 1) * ux - ux * 0.20))
                cy0 = int(round(r * uy + uy * 0.20))
                cy1 = int(round((r + 1) * uy - uy * 0.20))
                cell_sample = mask[cy0:max(cy0+1, cy1), cx0:max(cx0+1, cx1)]
                if cell_sample.size > 0 and cell_sample.mean() > 0.35:
                    mat[r, c] = 1

        # Cắt bỏ hàng/cột trống nếu có
        vr = np.any(mat == 1, axis=1)
        vc = np.any(mat == 1, axis=0)
        if not (np.any(vr) and np.any(vc)):
            results.append(None)
            continue

        cleaned_mat = mat[vr][:, vc].tolist()
        results.append({
            "matrix": cleaned_mat,
            "bbox": bbox,
            "confidence": 0.95
        })

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
# 4. THUẬT TOÁN GIẢI TỐI ƯU
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
# 5. VẼ ẢNH KẾT QUẢ
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
        d.line([(x, oy), (x, oy + bw)], fill=(255, 255, 255, 70), width=1)
        y = oy + k * cw
        d.line([(ox, y), (ox + bw, y)], fill=(255, 255, 255, 70), width=1)

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
                            fill=col + (140,), outline=col + (255,), width=3)

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

    # Chú thích góc phải
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
# 6. CÁC ĐƯỜNG DẪN (API ENDPOINTS)
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