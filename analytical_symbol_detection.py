#!/usr/bin/env python3
"""
- PDF → PNG (PyMuPDF)
- ROI interactive or --roi x,y,w,h
- Coarse-to-fine multi-scale (and optional multi-angle) template matching:
    Stage 1: run on downscaled image, keep top-K peaks per scale/angle
    Stage 2: refine around each peak at full resolution
- Exports YOLO labels for the FULL-RES image
"""

import argparse, json, os, math
from dataclasses import dataclass
from typing import List, Tuple

import cv2
import numpy as np

try:
    import fitz  # PyMuPDF
except Exception:
    fitz = None


# ----------------------------- Args ---------------------------------

@dataclass
class Args:
    pdf: str
    page: int
    zoom: float
    outdir: str
    threshold: float
    scales: List[float]
    angles: List[float]
    method: int
    class_name: str
    use_edges: bool
    roi: Tuple[int,int,int,int] | None
    reuse_roi: bool
    preview_width: int
    preview_height: int
    # Speed controls
    coarse: float
    topk: int
    refine_pad: float

def parse_args() -> Args:
    p = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    p.add_argument("--pdf", required=True, help="Path to input PDF")
    p.add_argument("--page", type=int, default=0, help="Zero-based page index")
    p.add_argument("--zoom", type=float, default=6.0, help="Rasterization zoom (higher => bigger image)")
    p.add_argument("--outdir", default="hvac_detect_out", help="Output directory")

    p.add_argument("--threshold", type=float, default=0.65, help="Template-match threshold (0..1)")
    p.add_argument("--scales", default="0.95,1.0,1.05", help="Comma-separated scales to search")
    p.add_argument("--angles", default="0", help="Comma-separated angles in degrees (e.g. 0,90). Default=0 (fast)")
    p.add_argument("--method", choices=["CCOEFF", "CCOEFF_NORMED", "SQDIFF", "SQDIFF_NORMED", "CCORR_NORMED"],
                   default="CCOEFF_NORMED", help="OpenCV matchTemplate method")

    p.add_argument("--class-name", default="object", help="YOLO class name")
    p.add_argument("--no-edges", dest="use_edges", action="store_false",
                   help="Match on grayscale instead of edges")

    p.add_argument("--roi", type=str, default=None, help="ROI as x,y,w,h (skip interactive selection)")
    p.add_argument("--reuse-roi", action="store_true", help="Reuse ROI saved in outdir/roi.json if present")
    p.add_argument("--preview-width", type=int, default=2400, help="Interactive ROI window max width")
    p.add_argument("--preview-height", type=int, default=1400, help="Interactive ROI window max height")

    p.add_argument("--coarse", type=float, default=0.5,
                   help="Downscale factor for Stage 1 (e.g., 0.5 = half-size). 0<coarse<=1")
    p.add_argument("--topk", type=int, default=300, help="Keep at most this many coarse candidates total")
    p.add_argument("--refine-pad", type=float, default=0.5,
                   help="Padding around template size during refinement (e.g., 0.5 = +50% in each dim)")

    a = p.parse_args()

    if fitz is None:
        raise RuntimeError("PyMuPDF (fitz) not found. Install with: pip install pymupdf")
    if not os.path.exists(a.pdf):
        raise FileNotFoundError(a.pdf)

    def parse_float_list(s: str) -> List[float]:
        out = []
        for tok in s.split(","):
            tok = tok.strip()
            if tok:
                out.append(float(tok))
        return out

    scales = parse_float_list(a.scales)
    angles = parse_float_list(a.angles)

    method_map = {
        "CCOEFF": cv2.TM_CCOEFF,
        "CCOEFF_NORMED": cv2.TM_CCOEFF_NORMED,
        "SQDIFF": cv2.TM_SQDIFF,
        "SQDIFF_NORMED": cv2.TM_SQDIFF_NORMED,
        "CCORR_NORMED": cv2.TM_CCORR_NORMED,
    }

    roi = None
    if a.roi:
        parts = [int(v) for v in a.roi.split(",")]
        if len(parts) != 4:
            raise ValueError("--roi must be x,y,w,h")
        roi = tuple(parts)

    coarse = float(a.coarse)
    if not (0 < coarse <= 1.0):
        raise ValueError("--coarse must be in (0,1]")

    return Args(
        pdf=a.pdf, page=a.page, zoom=a.zoom, outdir=a.outdir,
        threshold=a.threshold, scales=scales, angles=angles,
        method=method_map[a.method],
        class_name=a.class_name, use_edges=a.use_edges,
        roi=roi, reuse_roi=a.reuse_roi,
        preview_width=a.preview_width, preview_height=a.preview_height,
        coarse=coarse, topk=a.topk, refine_pad=a.refine_pad
    )


# ------------------------- PDF / IO utils --------------------------

def render_pdf_to_png(pdf_path: str, page_index: int, zoom: float, out_png: str):
    doc = fitz.open(pdf_path)
    if page_index < 0 or page_index >= len(doc):
        doc.close()
        raise IndexError(f"PDF has {len(doc)} pages; got page={page_index}")
    page = doc.load_page(page_index)
    mat = fitz.Matrix(zoom, zoom)
    pix = page.get_pixmap(matrix=mat, alpha=False)
    os.makedirs(os.path.dirname(out_png), exist_ok=True)
    pix.save(out_png)
    doc.close()

def ensure_image(pdf_path: str, page: int, zoom: float, outdir: str) -> str:
    out_png = os.path.join(outdir, f"page{page}.png")
    if not os.path.exists(out_png):
        print(f"[i] Rendering PDF → {out_png} (zoom={zoom})")
        render_pdf_to_png(pdf_path, page, zoom, out_png)
    else:
        print(f"[i] Using cached image: {out_png}")
    return out_png

def save_preview_images(img_bgr, outdir: str):
    gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
    edges = cv2.Canny(cv2.GaussianBlur(gray, (3, 3), 0), 50, 150)
    cv2.imwrite(os.path.join(outdir, "page_gray.png"), gray)
    cv2.imwrite(os.path.join(outdir, "page_edges.png"), edges)
    return gray, edges

def select_roi_interactive_scaled(img_bgr_full, max_w: int, max_h: int):
    Hf, Wf = img_bgr_full.shape[:2]
    scale = min(max_w / float(Wf), max_h / float(Hf), 1.0)
    if scale < 1.0:
        disp = cv2.resize(img_bgr_full, (int(Wf * scale), int(Hf * scale)), interpolation=cv2.INTER_AREA)
    else:
        disp = img_bgr_full.copy()
    cv2.namedWindow("Select ROI (preview)", cv2.WINDOW_NORMAL)
    rx, ry, rw, rh = cv2.selectROI("Select ROI (preview)", disp, showCrosshair=True, fromCenter=False)
    cv2.destroyAllWindows()
    if rw <= 0 or rh <= 0:
        raise RuntimeError("ROI selection canceled or invalid.")
    x = int(round(rx / scale)); y = int(round(ry / scale))
    w = int(round(rw / scale)); h = int(round(rh / scale))
    return x, y, w, h

def clamp_roi(x, y, w, h, W, H) -> Tuple[int, int, int, int]:
    x = max(0, min(W - 1, x))
    y = max(0, min(H - 1, y))
    w = max(8, min(W - x, w))
    h = max(8, min(H - y, h))
    return int(x), int(y), int(w), int(h)

def draw_box(img_bgr, box, color=(0, 0, 255), thick=2):
    x, y, w, h = box
    cv2.rectangle(img_bgr, (x, y), (x + w, y + h), color, thick)

def nms_xyxy(boxes: List[List[int]], scores: List[float], iou_thresh=0.3) -> List[int]:
    if not boxes:
        return []
    b = np.array(boxes, dtype=np.float32)
    s = np.array(scores, dtype=np.float32)
    x1, y1, x2, y2 = b.T
    areas = (x2 - x1 + 1) * (y2 - y1 + 1)
    order = s.argsort()[::-1]
    keep = []
    while order.size > 0:
        i = order[0]
        keep.append(i)
        xx1 = np.maximum(x1[i], x1[order[1:]])
        yy1 = np.maximum(y1[i], y1[order[1:]])
        xx2 = np.minimum(x2[i], x2[order[1:]])
        yy2 = np.minimum(y2[i], y2[order[1:]])
        w_ = np.maximum(0.0, xx2 - xx1 + 1)
        h_ = np.maximum(0.0, yy2 - yy1 + 1)
        inter = w_ * h_
        iou = inter / (areas[i] + areas[order[1:]] - inter + 1e-6)
        inds = np.where(iou <= iou_thresh)[0]
        order = order[inds + 1]
    return keep

def export_yolo(full_W, full_H, boxes_nms: List[List[int]], out_txt: str, class_id: int = 0):
    with open(out_txt, "w") as f:
        for (bx1, by1, bx2, by2) in boxes_nms:
            cx = (bx1 + bx2) / 2.0
            cy = (by1 + by2) / 2.0
            bw = (bx2 - bx1)
            bh = (by2 - by1)
            f.write(f"{class_id} {cx/full_W:.6f} {cy/full_H:.6f} {bw/full_W:.6f} {bh/full_H:.6f}\n")


def interior_mask_ccomp(
    tmpl_img: np.ndarray,
    use_otsu: bool = True,
    thresh_val: int = 127,
    invert: bool = True,
    close_ksize: int = 3,
    min_area_ratio: float = 0.003,   # ignore tiny specks (<0.3% of ROI)
    keep_mode: str = "largest",          # "all" | "largest" | "centered"
    center_ratio: float = 0.85       # for "centered": keep CCs whose centroid lies in the central box
) -> np.ndarray:
    """
    Build a mask (0/255) of ALL enclosed regions (holes) inside the symbol.
    Uses cv2.RETR_CCOMP so child contours are holes (parent != -1).

    Params:
      - use_otsu: use Otsu binarization (adapts threshold). If False, uses 'thresh_val'.
      - invert: set True when strokes are dark; we invert to make strokes=255.
      - close_ksize: morphology close kernel (>=1) to seal tiny gaps in strokes.
      - min_area_ratio: drop tiny enclosed regions by area relative to ROI.
      - keep_mode:
          "all"      -> union of all holes that pass filters
          "largest"  -> keep only the largest enclosed region
          "centered" -> keep holes whose centroid lies inside the central box (center_ratio * ROI)
    """
    # 1) Ensure single channel
    gray = cv2.cvtColor(tmpl_img, cv2.COLOR_BGR2GRAY) if tmpl_img.ndim == 3 else tmpl_img.copy()

    # 2) Binarize: strokes -> 255
    if use_otsu:
        _, bw = cv2.threshold(gray, 0, 255,
                              (cv2.THRESH_BINARY_INV if invert else cv2.THRESH_BINARY) + cv2.THRESH_OTSU)
    else:
        _, bw = cv2.threshold(gray, thresh_val, 255,
                              cv2.THRESH_BINARY_INV if invert else cv2.THRESH_BINARY)

    # 3) Seal small gaps so holes are truly enclosed
    if close_ksize and close_ksize > 0:
        k = np.ones((close_ksize, close_ksize), np.uint8)
        bw = cv2.morphologyEx(bw, cv2.MORPH_CLOSE, k, iterations=1)

    # 4) Find contours with two-level hierarchy (components + holes)
    contours, hierarchy = cv2.findContours(bw, cv2.RETR_CCOMP, cv2.CHAIN_APPROX_SIMPLE)
    H, W = gray.shape
    mask = np.zeros((H, W), np.uint8)
    if hierarchy is None or len(contours) == 0:
        return mask

    # Center box for the "centered" filter
    if keep_mode == "centered":
        cx1 = int((1 - center_ratio) / 2 * W)
        cy1 = int((1 - center_ratio) / 2 * H)
        cx2 = W - cx1
        cy2 = H - cy1

    # 5) Draw all hole contours (parent != -1) that pass filters
    roi_area = float(H * W)
    keep_idxs = []
    for i, cnt in enumerate(contours):
        parent = hierarchy[0][i][3]  # -1 = no parent (outer); >=0 = hole
        if parent == -1:
            continue  # skip outer strokes
        area = cv2.contourArea(cnt)
        if area < min_area_ratio * roi_area:
            continue  # tiny specks
        if keep_mode == "centered":
            m = cv2.moments(cnt)
            if m["m00"] == 0:
                continue
            cx = m["m10"] / m["m00"]
            cy = m["m01"] / m["m00"]
            if not (cx1 <= cx <= cx2 and cy1 <= cy <= cy2):
                continue
        keep_idxs.append(i)

    # Draw union
    for i in keep_idxs:
        cv2.drawContours(mask, contours, i, 255, thickness=-1)

    # Optionally keep only the largest connected component
    if keep_mode == "largest":
        num, labels, stats, _ = cv2.connectedComponentsWithStats(mask, connectivity=8)
        if num > 1:
            ii = 1 + np.argmax(stats[1:, cv2.CC_STAT_AREA])
            mask = np.where(labels == ii, 255, 0).astype(np.uint8)

    # Final clean
    if close_ksize and close_ksize > 0:
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, np.ones((3,3), np.uint8), iterations=1)

    return mask


def tighten_roi_with_mask(
    x: int, y: int, w: int, h: int,
    mask: np.ndarray,
    full_W: int, full_H: int,
    pad_px: int = 2,
    pad_ratio: float = 0.03,
    min_size: int = 8,
):
    """
    Given an initial ROI (x,y,w,h) and a binary mask created from that ROI crop
    (mask shape must be [h, w]), return a tightened ROI that is the axis-aligned
    bounding box of the mask (with optional padding).
    """
    if mask.ndim == 3:
        mask = mask[..., 0]
    ys, xs = np.where(mask > 0)
    if xs.size == 0 or ys.size == 0:
        # No foreground in mask -> keep original ROI
        return x, y, w, h

    # bbox in *ROI coordinates*
    x0 = int(xs.min()); y0 = int(ys.min())
    x1 = int(xs.max()); y1 = int(ys.max())

    # padding (pixel + relative to bbox)
    bw = x1 - x0 + 1; bh = y1 - y0 + 1
    pad = int(round(pad_px + pad_ratio * max(bw, bh)))

    # map to *page coordinates* + pad + clamp
    nx1 = max(0, x + x0 - pad)
    ny1 = max(0, y + y0 - pad)
    nx2 = min(full_W - 1, x + x1 + pad)
    ny2 = min(full_H - 1, y + y1 + pad)

    nw = max(min_size, nx2 - nx1 + 1)
    nh = max(min_size, ny2 - ny1 + 1)

    return int(nx1), int(ny1), int(nw), int(nh)

# ------------------------ Matching helpers -------------------------

def rotate_bound(img: np.ndarray, angle_deg: float) -> np.ndarray:
    """Rotate image keeping all content."""
    (h, w) = img.shape[:2]
    cX, cY = w // 2, h // 2
    M = cv2.getRotationMatrix2D((cX, cY), angle_deg, 1.0)
    cos = abs(M[0, 0]); sin = abs(M[0, 1])
    nW = int((h * sin) + (w * cos))
    nH = int((h * cos) + (w * sin))
    M[0, 2] += (nW / 2) - cX
    M[1, 2] += (nH / 2) - cY
    return cv2.warpAffine(img, M, (nW, nH), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REPLICATE)

def local_peaks(resmap: np.ndarray, thresh: float, max_peaks: int) -> List[Tuple[int,int,float]]:
    """Return (x,y,score) local maxima above thresh using dilation-based peak picking."""
    if resmap.size == 0:
        return []
    # normalize for SQDIFF* methods (lower is better → invert)
    inv = False
    peaks = []
    r = resmap
    if r.dtype != np.float32:
        r = r.astype(np.float32)
    # Find local maxima
    k = cv2.getStructuringElement(cv2.MORPH_RECT, (3,3))
    dil = cv2.dilate(r, k)
    mask = (r == dil) & (r >= thresh)
    ys, xs = np.where(mask)
    for y, x in zip(ys, xs):
        peaks.append((int(x), int(y), float(r[y, x])))
    peaks.sort(key=lambda t: t[2], reverse=True)
    return peaks[:max_peaks]

def match_stage_coarse(search_small: np.ndarray,
                       tmpl_base: np.ndarray,
                       scales: List[float],
                       angles: List[float],
                       method: int,
                       thresh: float,
                       max_keep: int,
                       shrink: float) -> List[Tuple[int,int,int,int,float,float,float]]:
    """
    Return a list of candidate boxes in FULL-RES coords:
    (x1,y1,x2,y2, score, scale, angle)
    """
    Hs, Ws = search_small.shape[:2]
    out = []
    for ang in angles:
        tmpl_rot = rotate_bound(tmpl_base, ang) if abs(ang) > 1e-3 else tmpl_base
        th0, tw0 = tmpl_rot.shape[:2]
        for s in scales:
            tw = max(8, int(round(tw0 * s * shrink)))
            th = max(8, int(round(th0 * s * shrink)))
            tmpl_rs = cv2.resize(tmpl_rot, (tw, th), interpolation=cv2.INTER_AREA)
            if tw >= Ws or th >= Hs:
                continue
            res = cv2.matchTemplate(search_small, tmpl_rs, method)
            # For SQDIFF methods, convert to similarity so we can use a single thresh direction
            if method in (cv2.TM_SQDIFF, cv2.TM_SQDIFF_NORMED):
                res_sim = 1.0 - res
                peaks = local_peaks(res_sim, thresh, max_keep)
            else:
                peaks = local_peaks(res, thresh, max_keep)
            for x, y, sc in peaks:
                # map back to FULL-RES coordinates
                x1 = int(round(x / shrink))
                y1 = int(round(y / shrink))
                x2 = int(round((x + tw) / shrink))
                y2 = int(round((y + th) / shrink))
                out.append((x1, y1, x2, y2, sc, s, ang))
    # keep global top-K by score
    out.sort(key=lambda t: t[4], reverse=True)
    return out[:max_keep]

def refine_one(full_img: np.ndarray,
               tmpl_full_rot_scaled: np.ndarray,
               box_hint: Tuple[int,int,int,int],
               method: int,
               pad_ratio: float,
               thresh: float) -> Tuple[int,int,int,int,float]:
    """Refine a coarse candidate around its hint box; return best (x1,y1,x2,y2,score)."""
    H, W = full_img.shape[:2]
    th, tw = tmpl_full_rot_scaled.shape[:2]
    x1h, y1h, x2h, y2h = box_hint
    # Expand by pad_ratio of template size
    pad_w = int(round(tw * pad_ratio))
    pad_h = int(round(th * pad_ratio))
    cx = (x1h + x2h) // 2
    cy = (y1h + y2h) // 2
    rx1 = max(0, cx - tw//2 - pad_w)
    ry1 = max(0, cy - th//2 - pad_h)
    rx2 = min(W, cx + tw//2 + pad_w)
    ry2 = min(H, cy + th//2 + pad_h)
    roi = full_img[ry1:ry2, rx1:rx2]
    if roi.shape[0] < th or roi.shape[1] < tw:
        return x1h, y1h, x2h, y2h, 0.0
    res = cv2.matchTemplate(roi, tmpl_full_rot_scaled, method)
    if method in (cv2.TM_SQDIFF, cv2.TM_SQDIFF_NORMED):
        min_val, max_val, min_loc, max_loc = cv2.minMaxLoc(res)
        sc = 1.0 - min_val
        best = min_loc
    else:
        min_val, max_val, min_loc, max_loc = cv2.minMaxLoc(res)
        sc = max_val
        best = max_loc
    if sc < thresh:
        return x1h, y1h, x2h, y2h, sc
    bx = best[0] + rx1
    by = best[1] + ry1
    return bx, by, bx + tw, by + th, sc


# ------------------------------- Main --------------------------------

def main():
    args = parse_args()
    os.makedirs(args.outdir, exist_ok=True)

    # 1) PDF → PNG
    page_png = ensure_image(args.pdf, args.page, args.zoom, args.outdir)
    img_bgr_full = cv2.imread(page_png)
    if img_bgr_full is None:
        raise RuntimeError(f"Failed to read {page_png}")
    Hf, Wf = img_bgr_full.shape[:2]

    # Previews
    gray_full = cv2.cvtColor(img_bgr_full, cv2.COLOR_BGR2GRAY)
    edges_full = cv2.Canny(cv2.GaussianBlur(gray_full, (3, 3), 0), 50, 150)
    cv2.imwrite(os.path.join(args.outdir, "page_gray.png"), gray_full)
    cv2.imwrite(os.path.join(args.outdir, "page_edges.png"), edges_full)

    # 2) ROI
    roi_path = os.path.join(args.outdir, "roi.json")
    if args.roi:
        x, y, w, h = args.roi
    elif args.reuse_roi and os.path.exists(roi_path):
        x, y, w, h = tuple(json.load(open(roi_path, "r")))
        print(f"[i] Reusing ROI from {roi_path}: {(x,y,w,h)}")
    else:
        x, y, w, h = select_roi_interactive_scaled(img_bgr_full.copy(), args.preview_width, args.preview_height)
    x, y, w, h = clamp_roi(x, y, w, h, Wf, Hf)
    json.dump([x, y, w, h], open(roi_path, "w"))
    print(f"[i] ROI: x={x}, y={y}, w={w}, h={h}  (saved to {roi_path})")
    
    annot = img_bgr_full.copy()
    cv2.rectangle(annot, (x, y), (x+w, y+h), (0, 0, 255), 3)
    cv2.imwrite(os.path.join(args.outdir, "annotated_roi_refined.png"), annot)

    search_full = gray_full
    first_gray_crop = gray_full[y:y+h, x:x+w].copy()

    cv2.imwrite(os.path.join(args.outdir, "template_1st_crop.png"), first_gray_crop)

    # Build interior-only mask 
    inner_mask = interior_mask_ccomp(
        tmpl_img=first_gray_crop,
        use_otsu=True,        
        invert=True,         
        close_ksize=3,        
        min_area_ratio=0.003,  
        keep_mode="all"    
    )

    # Tighten ROI to mask bounds + small padding
    x, y, w, h = tighten_roi_with_mask(
        x, y, w, h,
        inner_mask,
        full_W=Wf, full_H=Hf,
        pad_px=2, pad_ratio=0.03, 
        min_size=8
    )

    # TODO: if mask failed
    # Template crops
    if args.use_edges:
        search_full = edges_full
        tmpl_base_full = edges_full[y:y+h, x:x+w].copy()
    else:
        search_full = gray_full
        tmpl_base_full = gray_full[y:y+h, x:x+w].copy()
    
    cv2.imwrite(os.path.join(args.outdir, "template_adj_crop.png"), tmpl_base_full)

    print(f"[i] Coarse-to-fine matching on {'EDGES' if args.use_edges else 'GRAYSCALE'}")
    print(f"    scales={args.scales}  angles={args.angles}  method={args.method}  thresh={args.threshold}")
    print(f"    coarse={args.coarse}  topk={args.topk}  refine_pad={args.refine_pad}")

    # 3) Stage 1: COARSE search on downscaled page
    shrink = args.coarse
    if shrink < 1.0:
        search_small = cv2.resize(search_full, (int(Wf*shrink), int(Hf*shrink)), interpolation=cv2.INTER_AREA)
    else:
        search_small = search_full.copy()

    # Build coarse candidates
    coarse_cands = match_stage_coarse(
        search_small=search_small,
        tmpl_base=tmpl_base_full,
        scales=args.scales,
        angles=args.angles,
        method=args.method,
        thresh=args.threshold,
        max_keep=args.topk,
        shrink=shrink
    )
    print(f"[i] Coarse candidates kept: {len(coarse_cands)}")

    # 4) Stage 2: REFINE around each candidate at FULL resolution
    refined_boxes, refined_scores = [], []
    # Precompute rotated+scaled templates (full-res) to avoid re-rotating per cand
    rotated_scaled_cache: dict[tuple[float,float], np.ndarray] = {}
    for x1c, y1c, x2c, y2c, sc, s, ang in coarse_cands:
        key = (s, ang)
        if key not in rotated_scaled_cache:
            base_rot = rotate_bound(tmpl_base_full, ang) if abs(ang) > 1e-3 else tmpl_base_full
            th0, tw0 = base_rot.shape[:2]
            tw = max(8, int(round(tw0 * s)))
            th = max(8, int(round(th0 * s)))
            rotated_scaled_cache[key] = cv2.resize(base_rot, (tw, th), interpolation=cv2.INTER_AREA)
        tmpl_rs = rotated_scaled_cache[key]
        bx1, by1, bx2, by2, best_sc = refine_one(
            full_img=search_full,
            tmpl_full_rot_scaled=tmpl_rs,
            box_hint=(x1c, y1c, x2c, y2c),
            method=args.method,
            pad_ratio=args.refine_pad,
            thresh=args.threshold * 0.95  # slight relaxation in refine
        )
        refined_boxes.append([bx1, by1, bx2, by2])
        refined_scores.append(best_sc)

    # NMS at the end
    keep = nms_xyxy(refined_boxes, refined_scores, iou_thresh=0.25)
    boxes_nms = [refined_boxes[i] for i in keep]
    scores_nms = [refined_scores[i] for i in keep]
    print(f"[i] Final matches after refine+NMS: {len(boxes_nms)}")

    # 5) Visualization + YOLO export
    vis = img_bgr_full.copy()
    for (bx1, by1, bx2, by2), sc in zip(boxes_nms, scores_nms):
        cv2.rectangle(vis, (bx1, by1), (bx2, by2), (0,255,0), 2)
        cv2.putText(vis, f"{sc:.2f}", (bx1, max(10, by1 - 5)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0,255,0), 2)
    matches_png = os.path.join(args.outdir, "matches.png")
    cv2.imwrite(matches_png, vis)

    labels_txt = os.path.join(args.outdir, f"page{args.page}.txt")
    classes_txt = os.path.join(args.outdir, "classes.txt")
    export_yolo(Wf, Hf, boxes_nms, labels_txt, class_id=0)
    with open(classes_txt, "w") as f:
        f.write(args.class_name + "\n")

    print("\n=== Outputs ===")
    print(f"Rendered page:   {page_png}")
    print(f"Template crop:   {os.path.join(args.outdir, 'template_crop.png')}")
    print(f"Matches image:   {matches_png}")
    print(f"YOLO labels:     {labels_txt}")
    print(f"Class names:     {classes_txt}")
    print("================")
    if not boxes_nms:
        print("[!] If you missed matches, try lower --threshold (e.g., 0.50), increase --topk, or set --coarse 0.6.")
        print("For rotated symbols, add a few angles like --angles 0,90 (keeps fast).")
    # ------------------------------------------------------------------

if __name__ == "__main__":
    main()
