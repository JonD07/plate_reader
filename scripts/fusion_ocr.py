"""Prototype the fusion OCR: template NCC + tesseract ensemble + format prior.

Evaluated on zed_0.png (truth MZP9736) and the glyphs of the good image.
"""
import cv2
import numpy as np
import pytesseract
import re
from PIL import Image, ImageDraw, ImageFont

CHARS = 'ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789'
LETTERS = set('ABCDEFGHIJKLMNOPQRSTUVWXYZ')
DIGITS = set('0123456789')

FONT_PATHS = [
    '/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf',
    '/usr/share/fonts/truetype/liberation/LiberationSans-Bold.ttf',
]


def build_templates():
    tmpls_by_font = []
    for fp in FONT_PATHS:
        f = ImageFont.truetype(fp, 64)
        tmpls = {}
        for ch in CHARS:
            im = Image.new('L', (90, 90), 0)
            ImageDraw.Draw(im).text((12, 4), ch, font=f, fill=255)
            a = np.array(im)
            ys, xs = np.where(a > 40)
            if len(xs):
                tmpls[ch] = a[ys.min():ys.max() + 1, xs.min():xs.max() + 1]
        tmpls_by_font.append(tmpls)
    return tmpls_by_font


TMPLS = build_templates()


def ncc(A, B):
    a = A - A.mean()
    b = B - B.mean()
    d = np.sqrt((a * a).sum() * (b * b).sum())
    return (a * b).sum() / d if d > 1e-6 else 0.0


def segment_glyphs(band_gray):
    """X-projection segmentation: column runs of foreground pixels.

    At low resolution connected components fragment vertically, but the
    per-character column runs stay intact. Narrow runs (<3 px) are merged
    into an adjacent run (broken strokes). Returns [(x,y,w,h)].
    """
    th = cv2.threshold(band_gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)[1]
    th = cv2.morphologyEx(th, cv2.MORPH_OPEN, np.ones((2, 2), np.uint8))
    proj = (th > 0).sum(axis=0)
    runs = []
    in_run = False
    for i, v in enumerate(proj):
        if v > 0 and not in_run:
            start = i
            in_run = True
        elif v == 0 and in_run:
            runs.append([start, i])
            in_run = False
    if in_run:
        runs.append([start, len(proj)])
    # merge narrow fragments into the taller neighbor
    merged = []
    for r in runs:
        if merged and (r[1] - r[0] < 3 or merged[-1][1] - merged[-1][0] < 3) \
                and r[0] - merged[-1][1] <= 2:
            merged[-1][1] = r[1]
        else:
            merged.append(list(r))
    H = th.shape[0]
    glyphs = []
    for x0, x1 in merged:
        col = th[:, x0:x1]
        rows = np.where(col.sum(axis=1) > 0)[0]
        if len(rows) == 0:
            continue
        y0, y1 = rows[0], rows[-1] + 1
        h = y1 - y0
        if h < 0.15 * H or h > 0.95 * H:
            continue
        glyphs.append((x0, y0, x1 - x0, h))
    return glyphs, th


def template_scores(glyph_img):
    """Per-class NCC score (max over fonts). glyph_img: float 0..1, glyph tight."""
    h, w = glyph_img.shape
    scores = {}
    for tmpls in TMPLS:
        for ch, tt in tmpls.items():
            t2 = cv2.resize(tt.astype(np.float32), (w, h), interpolation=cv2.INTER_AREA) / 255.0
            s = ncc(t2, glyph_img)
            scores[ch] = max(scores.get(ch, -2.0), s)
    return scores


def tess_votes(band_gray):
    """Run tesseract variants; return list of reads (uppercase alnum only)."""
    cfg0 = '-c tessedit_char_whitelist=ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789'
    reads = []
    for scale in (5, 8):
        up = cv2.resize(band_gray, None, fx=scale, fy=scale, interpolation=cv2.INTER_LANCZOS4)
        variants = [('gray', up),
                    ('otsu', cv2.threshold(up, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)[1]),
                    ('sharp', cv2.addWeighted(up, 2.0, cv2.GaussianBlur(up, (0, 0), 3.0), -1.0, 0))]
        for name, prep in variants:
            prep = cv2.copyMakeBorder(prep, 30, 30, 60, 60, cv2.BORDER_REPLICATE)
            for psm in (7, 6):
                txt = pytesseract.image_to_string(prep, config=f'--psm {psm} {cfg0}').strip()
                t = re.sub(r'[^A-Z0-9]', '', txt.upper())
                if len(t) >= 4:
                    reads.append(t)
    return reads


# common US plate patterns by length (L=letter, N=digit) — conservative,
# formats actually issued by US states; an over-broad set lets garbage
# reads like 'N2P9736' (LNLNNNN) survive the format prior
PATTERNS = {
    5: ['LLNNN', 'LNNLL', 'NNLLL', 'LLLNN', 'NLLLL', 'LNNNN', 'NNNNL'],
    6: ['LLLNNN', 'NNNLLL', 'LLNNNN', 'NNNNLL', 'LNNNNN', 'NNLLLL'],
    7: ['LLLNNNN', 'NLLLNNN', 'LNNNNNN', 'NNNLLLL', 'NNNNLLL', 'LLNNNNN',
        'NNLLLLN', 'NNLLLLL'],
    8: ['LLNNNNNN', 'NNNNNLLL', 'LLLNNNNN', 'NNNLLLLN', 'NNLLLLNN',
        'LNNLLNNN'],
}


def p10_ensemble(band_gray, th, glyphs):
    """Per-glyph single-char OCR (psm10) across upscale/prep variants.

    Returns list of Counters (one per glyph) of per-class votes.
    """
    from collections import Counter
    out = []
    H = th.shape[0]
    for (x, y, w, h) in glyphs:
        g = 255 - th[max(0, y - 2):y + h + 2, max(0, x - 2):x + w + 2]
        votes = Counter()
        for target_h in (32, 48, 64):
            f = target_h / max(h, 1)
            big = cv2.resize(g, None, fx=f, fy=f, interpolation=cv2.INTER_LANCZOS4)
            preps = [('raw', big),
                     ('bil', cv2.bilateralFilter(big, 9, 60, 60)),
                     ('otsu', cv2.threshold(big, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)[1])]
            for name, prep in preps:
                p = cv2.copyMakeBorder(prep, 25, 25, 25, 25, cv2.BORDER_CONSTANT, value=255)
                r = pytesseract.image_to_data(p, config='--psm 10 -c tessedit_char_whitelist=' + CHARS,
                                              output_type=pytesseract.Output.DICT)
                ch = ''.join(t.strip() for t in r['text'] if t.strip())
                if len(ch) == 1:
                    votes[ch] += 1
        out.append(votes)
    return out


def fusion_read(band_gray):
    """Triple-source fusion decode: template NCC + whole-line votes +
    per-glyph psm10 votes, then US format prior. Returns (text, conf)."""
    # supersample tiny bands before thresholding: smoother glyph masks
    if band_gray.shape[0] < 40:
        band_gray = cv2.resize(band_gray, None, fx=4, fy=4,
                               interpolation=cv2.INTER_LANCZOS4)
    glyphs, th = segment_glyphs(band_gray)
    if len(glyphs) < 4:
        return '', 0.0
    n = len(glyphs)
    # source 1: template NCC per glyph
    t_scores = []
    for (x, y, w, h) in glyphs:
        sub = (th[y:y + h, x:x + w] > 0).astype(np.float32)
        t_scores.append(template_scores(sub))
    # source 2: whole-line reads on a recomposed clean band — the raw band
    # often contains border/frame fragments that shift OCR's character
    # count; pasting the segmented glyphs onto a clean canvas avoids that
    H, W = th.shape
    canvas = np.full((H + 8, W + 8), 255, np.uint8)
    for (x, y, w, h) in glyphs:
        canvas[y + 4:y + 4 + h, x + 4:x + 4 + w] = 255 - th[y:y + h, x:x + w]
    reads = tess_votes(canvas)
    votes = [dict() for _ in range(n)]
    for r in reads:
        if len(r) == n:
            for i, c in enumerate(r):
                votes[i][c] = votes[i].get(c, 0) + 1
    # source 3: per-glyph psm10 ensemble
    p10 = p10_ensemble(band_gray, th, glyphs)
    # normalize each source to max=1 per position, then sum with equal weight
    combined = []
    for i in range(n):
        sc = {}
        for c in CHARS:
            tmax = max(t_scores[i].values()) + 1e-6
            tn = max(t_scores[i].get(c, 0.0), 0.0) / tmax
            vmax = max(votes[i].values()) if votes[i] else 0
            vn = votes[i].get(c, 0) / vmax if vmax else 0.0
            pmax = max(p10[i].values()) if p10[i] else 0
            pn = p10[i].get(c, 0) / pmax if pmax else 0.0
            sc[c] = tn + vn + pn
        combined.append(sc)
    # unconstrained decode
    uncon = ''.join(max(combined[i], key=combined[i].get) for i in range(n))

    def matches_pattern(t):
        if len(t) not in PATTERNS:
            return False
        return any(all((c in LETTERS) == (k == 'L') for c, k in zip(t, pat))
                   for pat in PATTERNS[len(t)])

    # format-constrained decode: a US pattern prior. If the unconstrained
    # text already matches a known pattern, keep it; otherwise take the
    # best pattern-constrained decode (max over a subset can never beat
    # the unconstrained argmax, so compare only among patterns)
    best_txt = uncon
    if not matches_pattern(uncon):
        best_s = -1.0
        for pat in PATTERNS.get(n, []):
            s = 0.0
            txt = ''
            for i, k in enumerate(pat):
                pool = LETTERS if k == 'L' else DIGITS
                c, v = max(((c, v) for c, v in combined[i].items() if c in pool), key=lambda t: t[1])
                s += v
                txt += c
            if s > best_s:
                best_txt, best_s = txt, s
        if best_s < 0:  # no pattern for this length
            best_s = sum(max(combined[i].values()) for i in range(n))
    else:
        best_s = sum(max(combined[i].values()) for i in range(n))
    conf = best_s / n / 3.0  # mean per-position normalized score, 0..1
    return best_txt, float(min(conf, 1.0))


if __name__ == '__main__':
    truth = 'MZP9736'
    img = cv2.imread('/home/z/my-project/upload/zed_0.png')
    # band = tight char row (native res)
    band = cv2.cvtColor(img[238:265, 630:702], cv2.COLOR_BGR2GRAY)
    txt, conf = fusion_read(band)
    print('zed_0 fusion:', txt, round(conf, 3), 'match', sum(a == b for a, b in zip(txt, truth)))
    # sanity: the good image, native-res crop of its plate
    img2 = cv2.imread('/home/z/my-project/upload/zed_1794941012026862333.png')
