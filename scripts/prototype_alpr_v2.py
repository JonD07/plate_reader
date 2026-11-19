"""Robust offline ALPR pipeline v2 (Python reference for the C++ node).

Pipeline per image:
  1. Localization: Sobel-x edges + morphological close -> contours scored by
     aspect ratio, fill, edge density, brightness. Perspective-correct.
  2. Character-band extraction: threshold the warped plate, find connected
     components, keep plate-sized glyphs, crop tightly, re-warp to canonical.
  3. OCR: bilateral filter, LANCZOS 5x upscale, tight crop, psm 7 whitelist.
  4. State: OCR top strip, fuzzy match US state names.
  5. Car color: k-means dominant color of body region above the plate.
"""
import cv2
import numpy as np
import re
import sys

US_STATES = [
    "ALABAMA", "ALASKA", "ARIZONA", "ARKANSAS", "CALIFORNIA", "COLORADO",
    "CONNECTICUT", "DELAWARE", "FLORIDA", "GEORGIA", "HAWAII", "IDAHO",
    "ILLINOIS", "INDIANA", "IOWA", "KANSAS", "KENTUCKY", "LOUISIANA",
    "MAINE", "MARYLAND", "MASSACHUSETTS", "MICHIGAN", "MINNESOTA",
    "MISSISSIPPI", "MISSOURI", "MONTANA", "NEBRASKA", "NEVADA",
    "NEW HAMPSHIRE", "NEW JERSEY", "NEW MEXICO", "NEW YORK",
    "NORTH CAROLINA", "NORTH DAKOTA", "OHIO", "OKLAHOMA", "OREGON",
    "PENNSYLVANIA", "RHODE ISLAND", "SOUTH CAROLINA", "SOUTH DAKOTA",
    "TENNESSEE", "TEXAS", "UTAH", "VERMONT", "VIRGINIA", "WASHINGTON",
    "WEST VIRGINIA", "WISCONSIN", "WYOMING", "DISTRICT OF COLUMBIA",
]

COLOR_NAMES = [
    ((255, 255, 255), "white"),   ((230, 230, 230), "white"),
    ((0, 0, 0), "black"),         ((40, 40, 40), "black"),
    ((128, 128, 128), "gray"),    ((165, 165, 165), "silver"),
    ((0, 0, 255), "red"),         ((160, 82, 45), "red"),
    ((0, 100, 180), "orange"),    ((0, 165, 255), "orange"),
    ((0, 200, 255), "yellow"),    ((35, 180, 200), "gold"),
    ((0, 255, 0), "green"),       ((80, 120, 60), "green"),
    ((255, 0, 0), "blue"),        ((130, 70, 20), "blue"),
    ((255, 0, 255), "magenta"),   ((200, 130, 180), "pink"),
    ((140, 100, 60), "brown"),    ((101, 67, 33), "brown"),
    ((255, 255, 0), "cyan"),      ((128, 128, 0), "teal"),
    ((80, 60, 40), "beige"),      ((111, 111, 140), "tan"),
    ((200, 200, 200), "white"),   ((50, 50, 60), "black"),
    ((60, 60, 80), "navy"),       ((90, 90, 110), "gray"),
]


def classify_color(bgr):
    b, g, r = [float(v) for v in bgr]
    v = max(b, g, r)
    s = (max(b, g, r) - min(b, g, r)) / max(v, 1e-6)
    if v < 45:
        return "black"
    if s < 0.12:
        return "white" if v > 190 else ("silver" if v > 130 else "gray")
    best, bd = None, 1e18
    for ref, name in COLOR_NAMES:
        d = sum((a - b) ** 2 for a, b in zip((b, g, r), ref))
        if d < bd:
            best, bd = name, d
    return best


def order_corners(pts):
    pts = np.array(pts, dtype=np.float32).reshape(4, 2)
    s = pts.sum(axis=1)
    d = np.diff(pts, axis=1).flatten()  # d = y - x
    return np.array([pts[np.argmin(s)],   # tl: min(x+y)
                     pts[np.argmin(d)],   # tr: min(y-x)
                     pts[np.argmax(s)],   # br: max(x+y)
                     pts[np.argmax(d)]],  # bl: max(y-x)
                    dtype=np.float32)


def find_plate_candidates(img):
    """Return scored list of (score, quad, bbox) plate candidates.

    Sobel-x closes into text blobs; same-row blobs are merged into text
    bands; each band is expanded to plate proportions (2.4:1) and scored.
    Final verification happens later via OCR (read_plate tries the top N).
    """
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    H, W = gray.shape
    # Sobel-x emphasizes vertical character strokes
    sobel = cv2.Sobel(gray, cv2.CV_8U, 1, 0, ksize=3)
    _, sobel_t = cv2.threshold(sobel, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    close = cv2.morphologyEx(sobel_t, cv2.MORPH_CLOSE,
                             cv2.getStructuringElement(cv2.MORPH_RECT, (23, 7)))
    contours, _ = cv2.findContours(close, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)

    boxes = []
    for c in contours:
        x, y, w, h = cv2.boundingRect(c)
        if w < 50 or h < 14 or h > H // 2:
            continue
        boxes.append([x, y, w, h])

    # merge same-row boxes (vertical overlap, small horizontal gap);
    # keep BOTH raw and merged boxes as seeds — merging helps plates whose
    # text splits into groups, raw survives images where merging over-reaches
    seeds = [list(b) for b in boxes]
    work = [list(b) for b in boxes]
    merged = True
    while merged:
        merged = False
        for i in range(len(work)):
            for j in range(i + 1, len(work)):
                a, b = work[i], work[j]
                vy = min(a[1] + a[3], b[1] + b[3]) - max(a[1], b[1])
                if vy < 0.5 * min(a[3], b[3]):
                    continue
                gap = max(b[0] - (a[0] + a[2]), a[0] - (b[0] + b[2]), 0)
                if gap > 1.2 * max(a[3], b[3]):
                    continue
                nx, ny = min(a[0], b[0]), min(a[1], b[1])
                nx2 = max(a[0] + a[2], b[0] + b[2])
                ny2 = max(a[1] + a[3], b[1] + b[3])
                work[i] = [nx, ny, nx2 - nx, ny2 - ny]
                work.pop(j)
                merged = True
                break
            if merged:
                break
    seeds.extend(work)

    out = []
    seen = set()
    # extra seeds: the image's bright plate-shaped regions — plates are
    # retroreflective, so the bright rectangle IS the plate; localization
    # merges can miss a plate split by wide internal character spaces
    bt = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)[1]
    bt = cv2.morphologyEx(bt, cv2.MORPH_CLOSE,
                          cv2.getStructuringElement(cv2.MORPH_RECT, (5, 3)))
    bcnts, _ = cv2.findContours(bt, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    bright_seeds = []
    for c in bcnts:
        bx, by, bw, bh = cv2.boundingRect(c)
        ar = bw / float(bh)
        if bw < 0.12 * W or not (1.6 <= ar <= 4.5):
            continue
        if cv2.contourArea(c) < 0.35 * bw * bh:
            continue
        bright_seeds.append((bx, by, bw, bh))
        seeds.append([bx, by, bw, bh])
    for (x, y, w, h) in seeds:
        # text-band filters: relaxed AR, decent size
        ar = w / float(h)
        if not (1.5 <= ar <= 8.0) or w < 80:
            continue
        # edge density inside (characters); bright-rectangle seeds already
        # passed a strict shape+fill test, so they get a lower bar (their
        # box includes the empty plate margin around the characters)
        is_bright = (x, y, w, h) in bright_seeds
        roi = sobel[y:y + h, x:x + w]
        edge_density = float(np.count_nonzero(roi > 80)) / (w * h)
        if edge_density < (0.04 if is_bright else 0.06):
            continue
        # candidate 1: the seed itself if already plate-proportioned
        if 1.8 <= ar <= 5.0:
            key = (x, y, w, h)
            if key not in seen:
                seen.add(key)
                out.append((1.0 + 2.0 * min(edge_density, 0.4), (x, y, w, h)))
        # candidates 2-4: expand around the seed center to plate proportions;
        # multiple scales handle seeds that cover only part of the plate
        # (e.g. when the plate number has a wide internal space)
        for f in (1.0, 1.6, 2.2):
            plate_h = int(max(h * 1.9, w * f / 2.4))
            plate_w = int(max(w * f, plate_h * 2.4))
            if plate_w > 1.5 * W or plate_h > H:
                continue
            cx_, cy_ = x + w / 2.0, y + h / 2.0
            py0 = int(cy_ - plate_h * 0.42)
            px0 = int(cx_ - plate_w / 2.0)
            bx = max(0, min(px0, W - plate_w))
            by = max(0, min(py0, H - plate_h))
            key = (bx, by, plate_w, plate_h)
            if key in seen:
                continue
            seen.add(key)
            score = 1.0 + 2.0 * min(edge_density, 0.4)
            out.append((score, (bx, by, plate_w, plate_h)))
    out.sort(key=lambda t: -t[0])
    return out


def extract_quad(c, bbox):
    """Best-4-corners of a contour: extreme points, then minAreaRect fallback.

    Never trusts approxPolyDP point count; guarantees a valid convex quad.
    """
    pts = c.reshape(-1, 2).astype(np.float32)
    s = pts.sum(axis=1)
    d = np.diff(pts, axis=1).flatten()  # d = y - x
    quad = np.array([pts[np.argmin(s)],  # tl: min(x+y)
                     pts[np.argmin(d)],  # tr: min(y-x)
                     pts[np.argmax(s)],  # br: max(x+y)
                     pts[np.argmax(d)]],  # bl: max(y-x)
                    dtype=np.float32)
    # validate: each corner must be a distinct point and quad must have area
    if len(np.unique(quad, axis=0)) == 4 and quad_area(quad) > 0.5 * bbox[2] * bbox[3] * 0.3:
        return quad
    # fallback: min area rectangle
    (cx, cy), (rw, rh), ang = cv2.minAreaRect(c)
    if rw < rh:  # normalize orientation
        (cx, cy), (rw, rh), ang = (cx, cy), (rh, rw), ang + 90
    # build corners by rotating the axis-aligned rect
    a = np.deg2rad(ang)
    ux, uy = np.cos(a), np.sin(a)   # long axis unit vector
    vx, vy = -np.sin(a), np.cos(a)  # short axis unit vector
    hw, hh = rw / 2.0, rh / 2.0
    corners = [np.array([cx - hw * ux - hh * vx, cy - hw * uy - hh * vy]),
               np.array([cx + hw * ux - hh * vx, cy + hw * uy - hh * vy]),
               np.array([cx + hw * ux + hh * vx, cy + hw * uy + hh * vy]),
               np.array([cx - hw * ux + hh * vx, cy - hw * uy + hh * vy])]
    return order_corners(np.array(corners, dtype=np.float32))


def quad_area(quad):
    x = quad[:, 0]
    y = quad[:, 1]
    return 0.5 * abs(np.dot(x, np.roll(y, 1)) - np.dot(np.roll(x, 1), y))


def warp_plate(img, quad, out_w=None, out_h=None):
    """Perspective-correct to a canonical plate, sized to the SOURCE.

    The output resolution tracks the quad's width (~1.5x, clamped to
    120..400 px): warping a 75-px plate to a fixed 400x200 canvas only
    interpolates (never adds) detail and blurs glyphs beyond OCR repair.
    """
    if out_w is None:
        top = np.linalg.norm(quad[1] - quad[0])
        bot = np.linalg.norm(quad[2] - quad[3])
        src_w = max((top + bot) / 2.0, 1.0)
        out_w = int(min(400, max(120, 1.5 * src_w)))
    if out_h is None:
        out_h = out_w // 2
    dst = np.float32([[0, 0], [out_w - 1, 0], [out_w - 1, out_h - 1], [0, out_h - 1]])
    M = cv2.getPerspectiveTransform(quad, dst)
    return cv2.warpPerspective(img, M, (out_w, out_h))


def char_band(warped):
    """Find the character band inside the warped plate.

    Returns the tightly cropped character image, or None.
    """
    g = cv2.cvtColor(warped, cv2.COLOR_BGR2GRAY)
    # characters are darker than plate bg on US plates (mostly)
    th = cv2.threshold(g, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)[1]
    th = cv2.morphologyEx(th, cv2.MORPH_OPEN, np.ones((2, 2), np.uint8))
    n, labels, stats, cents = cv2.connectedComponentsWithStats(th, 8)
    H, W = th.shape
    glyphs = []
    for i in range(1, n):
        x, y, w, h, a = stats[i]
        # plate glyphs: tall-ish, not full-width, mid-band
        if a < W * H * 0.0005 or h < H * 0.15 or h > H * 0.7 or w > W * 0.3:
            continue
        cy = y + h / 2.0
        if not (H * 0.15 < cy < H * 0.95):
            continue
        glyphs.append((x, y, w, h))
    if len(glyphs) < 4:
        return None
    glyphs.sort(key=lambda t: t[0])
    # cluster glyphs by vertical center into rows (state banner vs plate
    # number vs slogan); keep the row with the largest total glyph area —
    # plate number characters dominate
    rows = []  # each: [cys, total_area, glyph_indices]
    for idx, (x, y, w, h) in enumerate(glyphs):
        cy = y + h / 2.0
        placed = False
        for row in rows:
            if abs(row[0] / row[2] - cy) < 0.35 * h:
                row[0] += cy
                row[1] += w * h
                row[2] += 1
                row[3].append(idx)
                placed = True
                break
        if not placed:
            rows.append([cy, w * h, 1, [idx]])
    best_row = max(rows, key=lambda r: r[1])
    glyphs = [glyphs[i] for i in best_row[3]]
    if len(glyphs) < 4:
        return None
    glyphs.sort(key=lambda t: t[0])
    # drop leading/trailing glyphs separated from the character group by a
    # gap much wider than the widest INTERNAL gap (background blobs and
    # plate decorations sit far outside; genuine plate spaces are internal
    # and narrower than background separations). A gap only disqualifies an
    # edge glyph that is ALSO much smaller than the group's median glyph —
    # spaced plate formats ("BLUE 777") keep their full-width groups intact
    if len(glyphs) > 4:
        median_area = np.median([g[2] * g[3] for g in glyphs])
        gaps = [glyphs[i + 1][0] - (glyphs[i][0] + glyphs[i][2])
                for i in range(len(glyphs) - 1)]
        sorted_gaps = sorted(gaps)
        # widest plausible internal space: median gap * 8, capped at 25% W
        space_limit = min(8 * sorted_gaps[len(gaps) // 2], 0.25 * W)
        edge_limit = max(space_limit, 30)
        while len(glyphs) > 4:
            lg = glyphs[1][0] - (glyphs[0][0] + glyphs[0][2])
            tg = glyphs[-1][0] - (glyphs[-2][0] + glyphs[-2][2])
            lead_small = glyphs[0][2] * glyphs[0][3] < 0.35 * median_area
            tail_small = glyphs[-1][2] * glyphs[-1][3] < 0.35 * median_area
            if lg > edge_limit and lead_small:
                glyphs.pop(0)
            elif tg > edge_limit and tail_small:
                glyphs.pop()
            else:
                break
    # horizontal bounds from surviving glyphs
    x0 = max(0, min(g[0] for g in glyphs) - 6)
    x1 = min(W, max(g[0] + g[2] for g in glyphs) + 6)
    # vertical bounds: median-based so border blobs (taller/offset) don't
    # drag the crop into the plate frame
    tops = sorted(g[1] for g in glyphs)
    bots = sorted(g[1] + g[3] for g in glyphs)
    y0 = max(0, tops[len(tops) // 2] - 6)
    y1 = min(H, bots[len(bots) // 2] + 6)
    if x1 - x0 < 0.3 * W or y1 - y0 < 8:
        return None
    return warped[y0:y1, x0:x1]


def ocr_characters(band_img):
    """Bilateral + LANCZOS 5x + psm7 whitelist OCR. Returns (text, conf)."""
    import pytesseract
    g = cv2.cvtColor(band_img, cv2.COLOR_BGR2GRAY)
    bil = cv2.bilateralFilter(g, 9, 60, 60)
    up = cv2.resize(bil, None, fx=5, fy=5, interpolation=cv2.INTER_LANCZOS4)
    cfg = ('-c tessedit_char_whitelist=ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789 '
           '--psm 7')
    d = pytesseract.image_to_data(up, config=cfg, output_type=pytesseract.Output.DICT)
    words = [(t.strip(), int(c)) for t, c in zip(d['text'], d['conf'])
             if t.strip() and int(c) >= 0]
    if not words:
        return '', 0.0
    text = ''.join(w for w, _ in words)
    conf = sum(c for _, c in words) / len(words) / 100.0
    text = re.sub(r'[^A-Z0-9]', '', text.upper())
    return text, conf


PLATE_RE = re.compile(r"^[A-Z0-9]{4,8}$")


def ocr_state(warped):
    """OCR top strip, fuzzy match to state names via edit distance.

    Returns (state, confidence, raw_text). Only reports a state when the
    best match ratio is >= 0.6; raw text is always returned for the human.
    """
    import pytesseract
    from difflib import SequenceMatcher
    H = warped.shape[0]
    best = (None, 0.0, '')
    for frac in (0.20, 0.25):
        strip = warped[0:int(H * frac), :]
        g = cv2.cvtColor(strip, cv2.COLOR_BGR2GRAY)
        bil = cv2.bilateralFilter(g, 9, 60, 60)
        up = cv2.resize(bil, None, fx=4, fy=4, interpolation=cv2.INTER_LANCZOS4)
        for psm in (7, 6):
            txt = pytesseract.image_to_string(up, config=f'--psm {psm}').strip().upper()
            txt = re.sub(r'[^A-Z ]', '', txt)
            txt = re.sub(r'\s+', ' ', txt).strip()
            if not txt:
                continue
            ratio, st = max(
                ((SequenceMatcher(None, txt, s).ratio(), s) for s in US_STATES),
                key=lambda t: t[0])
            if ratio > best[1]:
                best = (st, ratio, txt)
    state = best[0] if best[1] >= 0.6 else None
    return state, best[1], best[2]


def car_color(img, quad):
    """Dominant color of car body around the plate.

    Samples 4 regions (bumper below, trunk above, left, right of plate),
    computes each region's dominant k-means color, then blends them
    weighted by squared brightness so well-lit panels dominate and
    shadow gaps are suppressed.
    """
    if quad is None:
        return None
    (tl, tr, br, bl) = quad.astype(int)
    cx = int((tl[0] + tr[0] + br[0] + bl[0]) / 4)
    plate_top = min(tl[1], tr[1])
    plate_bot = max(br[1], bl[1])
    plate_left = min(tl[0], bl[0])
    plate_right = max(tr[0], br[0])
    H, W = img.shape[:2]
    plate_w = plate_right - plate_left

    def dom_color(x0, y0, x1, y1):
        x0, y0 = max(0, x0), max(0, y0)
        x1, y1 = min(W, x1), min(H, y1)
        if x1 - x0 < 20 or y1 - y0 < 20:
            return None
        roi = img[y0:y1, x0:x1].reshape(-1, 3).astype(np.float32)
        if len(roi) < 100:
            return None
        criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 10, 1.0)
        _, labels, centers = cv2.kmeans(roi, 3, None, criteria, 3, cv2.KMEANS_PP_CENTERS)
        counts = np.bincount(labels.flatten(), minlength=3)
        return centers[int(np.argmax(counts))]

    regions = [
        dom_color(cx - 160, plate_bot + 15, cx + 160, plate_bot + 120),   # bumper
        dom_color(cx - 120, plate_top - 70, cx + 120, plate_top - 8),     # trunk
        dom_color(plate_left - plate_w - 40, plate_top - 40,
                  plate_left - 20, plate_bot + 40),                       # left
        dom_color(plate_right + 20, plate_top - 40,
                  plate_right + plate_w + 40, plate_bot + 40),            # right
    ]
    regions = [r for r in regions if r is not None]
    if not regions:
        return None
    weights = np.array([max(float(np.mean(r)), 1.0) ** 2 for r in regions])
    blend = np.average(np.array(regions), axis=0, weights=weights)
    return classify_color(blend)


def bbox_to_quad(bbox):
    x, y, w, h = bbox
    return np.array([[x, y], [x + w, y], [x + w, y + h], [x, y + h]],
                    dtype=np.float32)


def refine_candidate(img, bbox):
    """Shrink a candidate box to the bright plate rectangle inside it.

    Plates are retroreflective: within a loose detection box the plate is
    the largest bright plate-shaped (AR 1.5-4.5) region. Localization
    seeds often include car body around the plate, which corrupts the
    warp proportions and the character band. Returns the refined bbox.
    """
    x, y, w, h = bbox
    H, W = img.shape[:2]
    x, y = max(0, x), max(0, y)
    w = min(w, W - x)
    h = min(h, H - y)
    roi = cv2.cvtColor(img[y:y + h, x:x + w], cv2.COLOR_BGR2GRAY)
    th = cv2.threshold(roi, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)[1]
    th = cv2.morphologyEx(th, cv2.MORPH_CLOSE,
                          cv2.getStructuringElement(cv2.MORPH_RECT, (5, 3)))
    cnts, _ = cv2.findContours(th, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    best = None
    for c in cnts:
        bx, by, bw, bh = cv2.boundingRect(c)
        if bw < 0.35 * w or bh < 0.25 * h:
            continue
        ar = bw / float(bh)
        if not (1.5 <= ar <= 4.5):
            continue
        if cv2.contourArea(c) < 0.35 * bw * bh:
            continue
        if best is None or bw * bh > best[2] * best[3]:
            best = (bx, by, bw, bh)
    if best is None:
        return bbox
    bx, by, bw, bh = best
    return (x + bx, y + by, bw, bh)


def read_plate(path_or_img, max_candidates=12):
    if isinstance(path_or_img, str):
        img = cv2.imread(path_or_img)
    else:
        img = path_or_img
    if img is None:
        return {"error": "cannot read image"}
    cands = find_plate_candidates(img)
    if not cands:
        return {"error": "no plate candidate found"}
    results = []
    for score, bbox in cands[:max_candidates]:
        bbox = refine_candidate(img, bbox)
        quad = bbox_to_quad(bbox)
        warped = warp_plate(img, quad)
        band = char_band(warped)
        if band is None:
            continue
        text, conf = ocr_characters(band)
        if not PLATE_RE.match(text):
            # fusion OCR: recovers far/small plates whose glyphs are at
            # the resolution limit, where single-shot tesseract fails
            try:
                from fusion_ocr import fusion_read
                g = cv2.cvtColor(band, cv2.COLOR_BGR2GRAY)
                text2, conf2 = fusion_read(g)
            except Exception:
                text2, conf2 = '', 0.0
            if PLATE_RE.match(text2):
                text, conf = text2, conf2
        if PLATE_RE.match(text):
            results.append((text, conf, score, quad, warped))
    if not results:
        return {"error": "OCR failed on all candidates", "debug":
                {"n_candidates": len(cands)}}
    # prefer plates that look like real US formats, reject state banners
    def quality(t, c):
        from difflib import SequenceMatcher
        # US plates are 5-8 chars; 4 or fewer is almost always truncation
        fmt = 2.0 if 5 <= len(t) <= 8 else 0.25
        # a "plate" that fuzzy-matches a state name is the plate's banner
        # (e.g. NEW YORK), not the plate number
        st_ratio = max(SequenceMatcher(None, t, s).ratio() for s in US_STATES)
        banner = 0.1 if st_ratio > 0.75 else 1.0
        return fmt * banner * (0.5 + 0.5 * c)
    results.sort(key=lambda r: -quality(r[0], r[1]))
    text, conf, score, quad, warped = results[0]
    state, sconf, raw = ocr_state(warped)
    color = car_color(img, quad)
    return {
        "plate": text, "confidence": round(conf, 3),
        "state": state, "state_confidence": round(sconf, 3),
        "state_raw": raw, "car_color": color,
        "debug": {"candidate_score": round(score, 3), "bbox":
                  [int(v) for v in quad[0]] + [int(quad[2][0] - quad[0][0]),
                                               int(quad[2][1] - quad[0][1])],
                  "n_candidates": len(cands), "n_ocr_hits": len(results)},
    }


if __name__ == "__main__":
    import json
    for p in sys.argv[1:]:
        print(p.split('/')[-1], "->", json.dumps(read_plate(p)))
