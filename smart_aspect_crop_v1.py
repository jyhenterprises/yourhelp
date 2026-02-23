# smart_aspect_crop.py

# Crop images to a target aspect ratio WITHOUT resizing, while attempting to
# avoid clipping faces (and optionally bodies/animals using YOLO).

# - Input: folder of .jpg/.jpeg/.png (recursive optional)
# - Output: folder, same filenames (optional suffix), max-quality saves by default
# - CUDA: uses GPU automatically if torch detects CUDA

# Example usage
# Square crop, preserve names, add suffix _1x1, use CUDA automatically
# python smart_aspect_crop.py --input "C:\photos\in" --output "C:\photos\out" --ratio 1:1 --suffix _1x1 --use-yolo
# Instagram portrait 4:5, recurse into subfolders, preserve subdirs
# python smart_aspect_crop.py --input ./in --output ./out --ratio 4:5 --recursive --keep-subdirs --use-yolo
# Face-only (no YOLO), still CUDA-accelerated for face detection if available
# python smart_aspect_crop.py --input ./in --output ./out --ratio 16:9 --face-only
# Notes / behavior guarantees

# No shrinking / no resizing: the output dimensions are always ≤ original, because it crops only.

# Long side stays long side: it crops the dimension that violates the target ratio (width if too wide; height if too tall).

# Quality-first saving:

# JPEG saved at quality=100 and no chroma subsampling

# PNG saved with compress_level=0

# Face centering is best-effort; if a face is near the edge and the aspect ratio forces a cut, the code shifts the crop window as much as possible to keep faces inside.


from __future__ import annotations

import argparse
import math
import os
from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional, Tuple, Dict

import numpy as np
from PIL import Image, ImageOps
from tqdm import tqdm

# Face detection (GPU if available)
import torch
from facenet_pytorch import MTCNN

# Optional: YOLO for person/animal boxes (GPU if available)
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
    weight: float  # importance for center computation

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
    """
    Accepts:
      - "1:1", "4:5", "16:9"
      - "1.7777" (float string)
    Returns width/height ratio as float.
    """
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


def pil_to_rgb(pil_img: Image.Image) -> Image.Image:
    # Apply EXIF orientation safely so boxes match displayed image
    pil_img = ImageOps.exif_transpose(pil_img)
    if pil_img.mode != "RGB":
        pil_img = pil_img.convert("RGB")
    return pil_img


def compute_crop_dims(orig_w: int, orig_h: int, target_ratio: float) -> Tuple[int, int]:
    """
    Compute crop width/height that fits within original WITHOUT resizing,
    preserving orientation: crop either width or height depending on which side is too long.
    """
    img_ratio = orig_w / orig_h
    if math.isclose(img_ratio, target_ratio, rel_tol=1e-6, abs_tol=1e-6):
        return orig_w, orig_h

    if img_ratio > target_ratio:
        # Too wide: keep full height, crop width
        crop_h = orig_h
        crop_w = int(round(orig_h * target_ratio))
        crop_w = min(crop_w, orig_w)
    else:
        # Too tall: keep full width, crop height
        crop_w = orig_w
        crop_h = int(round(orig_w / target_ratio))
        crop_h = min(crop_h, orig_h)

    # Ensure at least 1 px
    crop_w = max(1, crop_w)
    crop_h = max(1, crop_h)
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


def adjust_to_include_union(
    x0: float,
    y0: float,
    crop_w: int,
    crop_h: int,
    img_w: int,
    img_h: int,
    important_boxes: List[Box],
) -> Tuple[float, float]:
    """
    Try to shift the crop window (without resizing) so the union of important boxes
    lies inside the crop, when possible.
    """
    if not important_boxes:
        return x0, y0

    # Union bounds
    ux1 = min(b.x1 for b in important_boxes)
    uy1 = min(b.y1 for b in important_boxes)
    ux2 = max(b.x2 for b in important_boxes)
    uy2 = max(b.y2 for b in important_boxes)

    # Current window
    x1 = x0
    y1 = y0
    x2 = x0 + crop_w
    y2 = y0 + crop_h

    # Shift to include union bounds if possible
    # If union is larger than crop in a dimension, we can't fully include it.
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

    # Clamp to image bounds
    x0 = max(0.0, min(x0, img_w - crop_w))
    y0 = max(0.0, min(y0, img_h - crop_h))
    return x0, y0


def crop_rect_from_center(
    img_w: int,
    img_h: int,
    crop_w: int,
    crop_h: int,
    center: Tuple[float, float],
) -> Tuple[int, int, int, int]:
    cx, cy = center
    x0 = cx - crop_w / 2.0
    y0 = cy - crop_h / 2.0
    x0 = max(0.0, min(x0, img_w - crop_w))
    y0 = max(0.0, min(y0, img_h - crop_h))
    return int(round(x0)), int(round(y0)), int(round(x0 + crop_w)), int(round(y0 + crop_h))


def detect_faces_mtcnn(mtcnn: MTCNN, pil_img_rgb: Image.Image) -> List[Box]:
    """
    Returns face boxes (x1,y1,x2,y2) in pixel coords.
    """
    # facenet-pytorch can take PIL directly
    boxes, probs = mtcnn.detect(pil_img_rgb)
    out: List[Box] = []
    if boxes is None:
        return out
    for i, b in enumerate(boxes):
        if b is None:
            continue
        p = float(probs[i]) if probs is not None and probs[i] is not None else 0.0
        # Filter weak detections a bit
        if p < 0.80:
            continue
        x1, y1, x2, y2 = map(float, b.tolist())
        out.append(Box(x1=x1, y1=y1, x2=x2, y2=y2, weight=3.0))
    return out


def detect_yolo_boxes(model: "YOLO", pil_img_rgb: Image.Image, conf: float) -> List[Box]:
    """
    Optional: detect persons + common animals.
    Uses YOLO's COCO classes (if using a COCO model).
    """
    # Convert to numpy RGB
    arr = np.array(pil_img_rgb)  # HWC RGB
    results = model.predict(arr, verbose=False, conf=conf)
    if not results:
        return []

    r0 = results[0]
    if r0.boxes is None or len(r0.boxes) == 0:
        return []

    # COCO class ids we care about:
    # person=0
    # dog=16, cat=15, horse=17, sheep=18, cow=19, elephant=20, bear=21, zebra=22, giraffe=23
    # (Depending on model/dataset, these ids match common COCO models.)
    keep_ids = {0, 15, 16, 17, 18, 19, 20, 21, 22, 23}

    out: List[Box] = []
    for b, c, cf in zip(r0.boxes.xyxy, r0.boxes.cls, r0.boxes.conf):
        cls_id = int(c.item())
        if cls_id not in keep_ids:
            continue
        x1, y1, x2, y2 = map(float, b.tolist())
        # Weight: persons/animals are helpful, but faces are higher priority
        wt = 1.5 if cls_id == 0 else 1.4
        out.append(Box(x1=x1, y1=y1, x2=x2, y2=y2, weight=wt))
    return out


def save_image_max_quality(pil_img_rgb: Image.Image, out_path: Path) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    ext = out_path.suffix.lower()

    if ext in (".jpg", ".jpeg"):
        # Max quality; avoid chroma subsampling
        pil_img_rgb.save(
            out_path,
            format="JPEG",
            quality=100,
            subsampling=0,
            optimize=False,
        )
    elif ext == ".png":
        # Avoid extra compression (faster + preserves exact pixels)
        pil_img_rgb.save(
            out_path,
            format="PNG",
            compress_level=0,
        )
    else:
        # Fallback
        pil_img_rgb.save(out_path)


def build_output_path(
    in_path: Path,
    input_root: Path,
    output_root: Path,
    suffix: str,
    keep_subdirs: bool,
) -> Path:
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
        img = Image.open(in_path)
        img = pil_to_rgb(img)
        w, h = img.size

        crop_w, crop_h = compute_crop_dims(w, h, target_ratio)

        img_ratio = w / h
        already = math.isclose(img_ratio, target_ratio, rel_tol=1e-6, abs_tol=1e-6)

        if already and skip_if_already_ratio:
            out_path = build_output_path(in_path, input_root, output_root, suffix, keep_subdirs)
            save_image_max_quality(img, out_path)
            return {"file": str(in_path), "status": "copied (already ratio)"}

        # Detect faces (GPU if available)
        face_boxes = [b.clamp(w, h) for b in detect_faces_mtcnn(mtcnn, img)]

        # Optionally detect persons/animals to reduce body clipping
        body_boxes: List[Box] = []
        if (not face_only) and (yolo_model is not None):
            body_boxes = [b.clamp(w, h) for b in detect_yolo_boxes(yolo_model, img, conf=yolo_conf)]

        # Compute center
        fallback_center = (w / 2.0, h / 2.0)
        all_boxes = face_boxes + body_boxes
        center = weighted_center(all_boxes, fallback=fallback_center)

        # Initial crop rect centered on weighted center
        x0, y0, x1, y1 = crop_rect_from_center(w, h, crop_w, crop_h, center)

        # Try to shift to fully include faces first; then include bodies if possible
        # (faces higher priority)
        x0f, y0f = adjust_to_include_union(x0, y0, crop_w, crop_h, w, h, face_boxes)
        x0b, y0b = adjust_to_include_union(x0f, y0f, crop_w, crop_h, w, h, body_boxes)
        x0, y0 = int(round(x0b)), int(round(y0b))
        x1, y1 = x0 + crop_w, y0 + crop_h

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
    ap.add_argument("--use-yolo", action="store_true", help="Use YOLO to bias cropping away from clipping bodies/animals (recommended).")
    ap.add_argument("--yolo-model", default="yolov8n.pt", type=str, help="YOLO model path/name (ultralytics). Default: yolov8n.pt")
    ap.add_argument("--yolo-conf", default=0.35, type=float, help="YOLO confidence threshold.")
    ap.add_argument("--skip-if-already-ratio", action="store_true", help="If image already matches ratio, just save/copy as-is.")
    ap.add_argument("--device", default="auto", choices=["auto", "cuda", "cpu"], help="Force device selection.")
    args = ap.parse_args()

    input_dir = Path(args.input).expanduser().resolve()
    output_dir = Path(args.output).expanduser().resolve()
    ensure_dir(output_dir)

    target_ratio = parse_aspect_ratio(args.ratio)

    # Device
    if args.device == "cuda":
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA requested but torch.cuda.is_available() is False.")
        device = torch.device("cuda:0")
    elif args.device == "cpu":
        device = torch.device("cpu")
    else:
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    # Face detector
    # MTCNN will use GPU if device is cuda.
    mtcnn = MTCNN(keep_all=True, device=device)

    # YOLO optional
    yolo_model = None
    if args.use_yolo:
        if not _HAVE_YOLO:
            raise RuntimeError("ultralytics not installed. Run: pip install ultralytics")
        yolo_model = YOLO(args.yolo_model)  # auto uses GPU if available

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

    # Summary
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