# smart_aspect_crop.py

# Crop images to a target aspect ratio WITHOUT resizing/shrinking.
# Rules:
#   1) Preserve the original LONG side dimension first (keep it at full length).
#   2) Crop the OTHER dimension by the MINIMUM number of pixels needed to hit the target ratio.
#   3) Place the crop to avoid clipping faces first; then bodies/animals if YOLO enabled.

# Color handling:
#   - NO color-space conversions unless unavoidable for output format.
#   - PNGs keep their original mode (RGBA stays RGBA).
#   - JPEGs are saved as JPEG with max quality; if source has alpha it is composited only if needed.

# python smart_aspect_crop.py --input ./images/in --output ./images/out --ratio 4:6 --recursive --keep-subdirs --use-yolo



from __future__ import annotations

import argparse
import math
from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional, Tuple, Dict

import numpy as np
from PIL import Image, ImageOps
from tqdm import tqdm

import torch
from facenet_pytorch import MTCNN

try:
    from ultralytics import YOLO
    _HAVE_YOLO = True
except Exception:
    _HAVE_YOLO = False

IMG_EXTS = {".jpg", ".jpeg", ".png"}


@dataclass
class Box:
    x1: float
    y1: float
    x2: float
    y2: float
    weight: float

    @property
    def cx(self) -> float:
        return (self.x1 + self.x2) / 2.0

    @property
    def cy(self) -> float:
        return (self.y1 + self.y2) / 2.0

    def clamp(self, w: int, h: int) -> "Box":
        return Box(
            x1=max(0.0, min(self.x1, w - 1)),
            y1=max(0.0, min(self.y1, h - 1)),
            x2=max(0.0, min(self.x2, w - 1)),
            y2=max(0.0, min(self.y2, h - 1)),
            weight=self.weight,
        )


def parse_aspect_ratio(s: str) -> float:
    s = s.strip().lower()
    if ":" in s:
        a, b = s.split(":")
        w = float(a)
        h = float(b)
        if w <= 0 or h <= 0:
            raise ValueError("Aspect ratio components must be > 0.")
        return w / h
    r = float(s)
    if r <= 0:
        raise ValueError("Aspect ratio must be > 0.")
    return r


def list_images(input_dir: Path, recursive: bool) -> List[Path]:
    if recursive:
        files = [p for p in input_dir.rglob("*") if p.suffix.lower() in IMG_EXTS]
    else:
        files = [p for p in input_dir.iterdir() if p.is_file() and p.suffix.lower() in IMG_EXTS]
    files.sort()
    return files


def ensure_dir(p: Path) -> None:
    p.mkdir(parents=True, exist_ok=True)


def load_image_no_color_change(path: Path) -> Image.Image:
    """
    Loads image and applies EXIF orientation. Does NOT convert color modes.
    """
    img = Image.open(path)
    img = ImageOps.exif_transpose(img)  # orientation only
    return img


def compute_crop_dims_preserve_long_side(orig_w: int, orig_h: int, target_ratio: float) -> Tuple[int, int]:
    """
    Preserve original long side first.
    - If landscape (w >= h): keep full width, crop height minimally to match ratio.
        ratio = w / crop_h  => crop_h = w / ratio
    - If portrait (h > w): keep full height, crop width minimally to match ratio.
        ratio = crop_w / h  => crop_w = ratio * h
    - If square: treat as landscape (keep width) unless ratio demands otherwise; still minimal crop.

    Ensures crop dims <= original dims, and at least 1px.
    """
    if orig_w <= 0 or orig_h <= 0:
        return 1, 1

    img_ratio = orig_w / orig_h
    if math.isclose(img_ratio, target_ratio, rel_tol=1e-9, abs_tol=1e-9):
        return orig_w, orig_h

    if orig_w >= orig_h:
        # Preserve width
        crop_w = orig_w
        crop_h = int(math.floor(orig_w / target_ratio + 1e-9))
        crop_h = min(crop_h, orig_h)
        # If numeric edge-case makes crop_h == orig_h but ratio still off, fallback to height-preserve
        if crop_h > orig_h:
            crop_h = orig_h
    else:
        # Preserve height
        crop_h = orig_h
        crop_w = int(math.floor(orig_h * target_ratio + 1e-9))
        crop_w = min(crop_w, orig_w)

    crop_w = max(1, min(crop_w, orig_w))
    crop_h = max(1, min(crop_h, orig_h))

    # If due to floor the ratio is slightly too far and we can still crop minimally by 1px, adjust:
    # We want crop dims that do not exceed original and preserve long side.
    # For landscape: crop_h should be as large as possible while w/crop_h <=? target? Actually we want exactly ratio:
    # Use round instead of floor but clamp safely:
    if orig_w >= orig_h:
        desired_h = int(round(orig_w / target_ratio))
        if 1 <= desired_h <= orig_h:
            crop_h = desired_h
    else:
        desired_w = int(round(orig_h * target_ratio))
        if 1 <= desired_w <= orig_w:
            crop_w = desired_w

    return crop_w, crop_h


def weighted_center(boxes: List[Box], fallback: Tuple[float, float]) -> Tuple[float, float]:
    if not boxes:
        return fallback
    wsum = sum(b.weight for b in boxes)
    if wsum <= 1e-9:
        return fallback
    cx = sum(b.cx * b.weight for b in boxes) / wsum
    cy = sum(b.cy * b.weight for b in boxes) / wsum
    return cx, cy


def crop_rect_from_center(img_w: int, img_h: int, crop_w: int, crop_h: int, center: Tuple[float, float]) -> Tuple[int, int, int, int]:
    cx, cy = center
    x0 = cx - crop_w / 2.0
    y0 = cy - crop_h / 2.0
    x0 = max(0.0, min(x0, img_w - crop_w))
    y0 = max(0.0, min(y0, img_h - crop_h))
    return int(round(x0)), int(round(y0)), int(round(x0 + crop_w)), int(round(y0 + crop_h))


def adjust_to_include_union(
    x0: float, y0: float,
    crop_w: int, crop_h: int,
    img_w: int, img_h: int,
    important_boxes: List[Box],
) -> Tuple[float, float]:
    if not important_boxes:
        return x0, y0

    ux1 = min(b.x1 for b in important_boxes)
    uy1 = min(b.y1 for b in important_boxes)
    ux2 = max(b.x2 for b in important_boxes)
    uy2 = max(b.y2 for b in important_boxes)

    x1 = x0
    y1 = y0
    x2 = x0 + crop_w
    y2 = y0 + crop_h

    if (ux2 - ux1) <= crop_w:
        if ux1 < x1:
            x0 -= (x1 - ux1)
        elif ux2 > x2:
            x0 += (ux2 - x2)

    if (uy2 - uy1) <= crop_h:
        if uy1 < y1:
            y0 -= (y1 - uy1)
        elif uy2 > y2:
            y0 += (uy2 - y2)

    x0 = max(0.0, min(x0, img_w - crop_w))
    y0 = max(0.0, min(y0, img_h - crop_h))
    return x0, y0


def detect_faces_mtcnn(mtcnn: MTCNN, pil_img_for_detect: Image.Image) -> List[Box]:
    """
    MTCNN expects RGB. For detection only, we convert a COPY to RGB if needed.
    This does NOT affect the saved output, which uses the original image mode.
    """
    if pil_img_for_detect.mode != "RGB":
        det_img = pil_img_for_detect.convert("RGB")
    else:
        det_img = pil_img_for_detect

    boxes, probs = mtcnn.detect(det_img)
    out: List[Box] = []
    if boxes is None:
        return out
    for i, b in enumerate(boxes):
        if b is None:
            continue
        p = float(probs[i]) if probs is not None and probs[i] is not None else 0.0
        if p < 0.80:
            continue
        x1, y1, x2, y2 = map(float, b.tolist())
        out.append(Box(x1=x1, y1=y1, x2=x2, y2=y2, weight=3.0))
    return out


def detect_yolo_boxes(model: "YOLO", pil_img_for_detect: Image.Image, conf: float) -> List[Box]:
    """
    YOLO expects RGB arrays; convert a COPY for detection only.
    """
    if pil_img_for_detect.mode != "RGB":
        det_img = pil_img_for_detect.convert("RGB")
    else:
        det_img = pil_img_for_detect

    arr = np.array(det_img)  # HWC RGB
    results = model.predict(arr, verbose=False, conf=conf)
    if not results:
        return []
    r0 = results[0]
    if r0.boxes is None or len(r0.boxes) == 0:
        return []

    keep_ids = {0, 15, 16, 17, 18, 19, 20, 21, 22, 23}
    out: List[Box] = []
    for b, c, cf in zip(r0.boxes.xyxy, r0.boxes.cls, r0.boxes.conf):
        cls_id = int(c.item())
        if cls_id not in keep_ids:
            continue
        x1, y1, x2, y2 = map(float, b.tolist())
        wt = 1.5 if cls_id == 0 else 1.4
        out.append(Box(x1=x1, y1=y1, x2=x2, y2=y2, weight=wt))
    return out


def save_image_max_quality(src_cropped: Image.Image, out_path: Path) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    ext = out_path.suffix.lower()

    if ext in (".jpg", ".jpeg"):
        # JPEG cannot store alpha; if present, composite onto black (or choose white via flag if you want).
        if src_cropped.mode in ("RGBA", "LA"):
            bg = Image.new("RGB", src_cropped.size, (0, 0, 0))
            bg.paste(src_cropped, mask=src_cropped.split()[-1])
            out_img = bg
        elif src_cropped.mode != "RGB":
            out_img = src_cropped.convert("RGB")
        else:
            out_img = src_cropped

        out_img.save(out_path, format="JPEG", quality=100, subsampling=0, optimize=False)

    elif ext == ".png":
        # Preserve mode exactly (RGBA stays RGBA)
        src_cropped.save(out_path, format="PNG", compress_level=0)

    else:
        src_cropped.save(out_path)


def build_output_path(in_path: Path, input_root: Path, output_root: Path, suffix: str, keep_subdirs: bool) -> Path:
    rel = in_path.relative_to(input_root) if keep_subdirs else Path(in_path.name)
    stem = rel.stem + (suffix or "")
    out_name = stem + rel.suffix
    return (output_root / rel.parent / out_name) if keep_subdirs else (output_root / out_name)


def process_one(
    in_path: Path,
    input_root: Path,
    output_root: Path,
    target_ratio: float,
    suffix: str,
    keep_subdirs: bool,
    mtcnn: MTCNN,
    yolo_model: Optional["YOLO"],
    yolo_conf: float,
    face_only: bool,
    skip_if_already_ratio: bool,
) -> Dict[str, str]:
    try:
        img = load_image_no_color_change(in_path)
        w, h = img.size
        img_ratio = w / h

        already = math.isclose(img_ratio, target_ratio, rel_tol=1e-9, abs_tol=1e-9)
        if already and skip_if_already_ratio:
            out_path = build_output_path(in_path, input_root, output_root, suffix, keep_subdirs)
            save_image_max_quality(img, out_path)
            return {"file": str(in_path), "status": "copied (already ratio)"}

        crop_w, crop_h = compute_crop_dims_preserve_long_side(w, h, target_ratio)

        face_boxes = [b.clamp(w, h) for b in detect_faces_mtcnn(mtcnn, img)]

        body_boxes: List[Box] = []
        if (not face_only) and (yolo_model is not None):
            body_boxes = [b.clamp(w, h) for b in detect_yolo_boxes(yolo_model, img, conf=yolo_conf)]

        fallback_center = (w / 2.0, h / 2.0)
        center = weighted_center(face_boxes + body_boxes, fallback=fallback_center)

        x0, y0, x1, y1 = crop_rect_from_center(w, h, crop_w, crop_h, center)

        # Shift to keep faces inside if possible, then bodies
        x0f, y0f = adjust_to_include_union(float(x0), float(y0), crop_w, crop_h, w, h, face_boxes)
        x0b, y0b = adjust_to_include_union(x0f, y0f, crop_w, crop_h, w, h, body_boxes)

        x0 = int(round(x0b))
        y0 = int(round(y0b))
        x1 = x0 + crop_w
        y1 = y0 + crop_h

        cropped = img.crop((x0, y0, x1, y1))

        out_path = build_output_path(in_path, input_root, output_root, suffix, keep_subdirs)
        save_image_max_quality(cropped, out_path)

        meta = "faces" if face_boxes else "no-faces"
        if body_boxes:
            meta += "+yolo"
        return {"file": str(in_path), "status": f"ok ({meta})"}

    except Exception as e:
        return {"file": str(in_path), "status": f"ERROR: {e}"}


def main() -> None:
    ap = argparse.ArgumentParser(description="CUDA-friendly smart aspect-ratio cropper (no resizing).")
    ap.add_argument("--input", required=True, type=str, help="Input folder of images.")
    ap.add_argument("--output", required=True, type=str, help="Output folder for cropped images.")
    ap.add_argument("--ratio", required=True, type=str, help='Target aspect ratio: e.g. "1:1", "4:5", "16:9", "0.8".')
    ap.add_argument("--suffix", default="", type=str, help='Optional suffix appended to filename stem (before extension).')
    ap.add_argument("--recursive", action="store_true", help="Recurse into subfolders.")
    ap.add_argument("--keep-subdirs", action="store_true", help="Preserve input subfolder structure in output.")
    ap.add_argument("--face-only", action="store_true", help="Use faces only (no YOLO person/animal bias).")
    ap.add_argument("--use-yolo", action="store_true", help="Use YOLO to bias cropping away from clipping bodies/animals.")
    ap.add_argument("--yolo-model", default="yolov8n.pt", type=str, help="YOLO model path/name (ultralytics).")
    ap.add_argument("--yolo-conf", default=0.35, type=float, help="YOLO confidence threshold.")
    ap.add_argument("--skip-if-already-ratio", action="store_true", help="If image already matches ratio, save as-is.")
    ap.add_argument("--device", default="auto", choices=["auto", "cuda", "cpu"], help="Force device selection.")
    args = ap.parse_args()

    input_dir = Path(args.input).expanduser().resolve()
    output_dir = Path(args.output).expanduser().resolve()
    ensure_dir(output_dir)

    target_ratio = parse_aspect_ratio(args.ratio)

    if args.device == "cuda":
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA requested but torch.cuda.is_available() is False.")
        device = torch.device("cuda:0")
    elif args.device == "cpu":
        device = torch.device("cpu")
    else:
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    mtcnn = MTCNN(keep_all=True, device=device)

    yolo_model = None
    if args.use_yolo:
        if not _HAVE_YOLO:
            raise RuntimeError("ultralytics not installed. Run: pip install ultralytics")
        yolo_model = YOLO(args.yolo_model)

    files = list_images(input_dir, recursive=args.recursive)
    if not files:
        print("No images found.")
        return

    results = []
    for p in tqdm(files, desc="Processing", unit="img"):
        res = process_one(
            in_path=p,
            input_root=input_dir,
            output_root=output_dir,
            target_ratio=target_ratio,
            suffix=args.suffix,
            keep_subdirs=args.keep_subdirs,
            mtcnn=mtcnn,
            yolo_model=yolo_model,
            yolo_conf=args.yolo_conf,
            face_only=args.face_only or (not args.use_yolo),
            skip_if_already_ratio=args.skip_if_already_ratio,
        )
        results.append(res)

    ok = sum(1 for r in results if r["status"].startswith("ok") or r["status"].startswith("copied"))
    err = len(results) - ok
    print(f"\nDone. {ok} succeeded, {err} failed.")
    if err:
        print("Failures:")
        for r in results:
            if r["status"].startswith("ERROR"):
                print(f' - {r["file"]}: {r["status"]}')


if __name__ == "__main__":
    main()