#!/usr/bin/env python3
# NOT WORKING

"""
smart_aspect_crop_jpegtran.py

- Crops images to a target aspect ratio WITHOUT resizing.
- Preserves the original LONG side dimension first, then crops minimum pixels
  on the other dimension to hit the target ratio.
- Chooses crop position to avoid clipping faces (MTCNN) and optionally
  persons/animals (YOLO).
- JPEG output uses jpegtran for LOSSLESS crop (no recompression => no color shift).

Important JPEG caveat:
- Lossless JPEG cropping requires MCU alignment. We use jpegtran -trim, which may
  shave a few pixels off. The resulting ratio may be slightly off by a tiny amount.

Dependencies:
  pip install pillow numpy tqdm torch facenet-pytorch
  pip install ultralytics  # optional for person/animal boxes
External:
  jpegtran must be installed and on PATH.
    Windows (recommended), using Powershell with "Run as administrator"
    Chocolatey: 
    choco install -y chocolatey-compatibility.extension
    choco install -y libjpeg-turbo --force
"""

from __future__ import annotations

import argparse
import math
import shutil
import subprocess
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


# --- Parsing / file helpers -------------------------------------------------

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


def ensure_dir(p: Path) -> None:
    p.mkdir(parents=True, exist_ok=True)


def list_images(input_dir: Path, recursive: bool) -> List[Path]:
    if recursive:
        files = [p for p in input_dir.rglob("*") if p.suffix.lower() in IMG_EXTS]
    else:
        files = [p for p in input_dir.iterdir() if p.is_file() and p.suffix.lower() in IMG_EXTS]
    files.sort()
    return files


def build_output_path(in_path: Path, input_root: Path, output_root: Path, suffix: str, keep_subdirs: bool) -> Path:
    rel = in_path.relative_to(input_root) if keep_subdirs else Path(in_path.name)
    stem = rel.stem + (suffix or "")
    out_name = stem + rel.suffix
    return (output_root / rel.parent / out_name) if keep_subdirs else (output_root / out_name)


# --- Crop sizing: preserve long side, crop minimum other side ---------------

def compute_crop_dims_preserve_long_side(orig_w: int, orig_h: int, target_ratio: float) -> Tuple[int, int]:
    """
    Preserve original long side dimension first, then crop minimum pixels on the other side.

    Landscape (w >= h): keep full width, crop height:
      target_ratio = w / crop_h  -> crop_h = w / target_ratio

    Portrait (h > w): keep full height, crop width:
      target_ratio = crop_w / h  -> crop_w = target_ratio * h
    """
    img_ratio = orig_w / orig_h
    if math.isclose(img_ratio, target_ratio, rel_tol=1e-9, abs_tol=1e-9):
        return orig_w, orig_h

    if orig_w >= orig_h:
        crop_w = orig_w
        crop_h = int(round(orig_w / target_ratio))
        crop_h = max(1, min(crop_h, orig_h))
    else:
        crop_h = orig_h
        crop_w = int(round(orig_h * target_ratio))
        crop_w = max(1, min(crop_w, orig_w))

    return crop_w, crop_h


# --- Crop placement utilities ----------------------------------------------

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


# --- Detection (RGB copies ONLY for detection; does not affect output) ------

def detect_faces_mtcnn(mtcnn: MTCNN, pil_img: Image.Image) -> List[Box]:
    det_img = pil_img.convert("RGB") if pil_img.mode != "RGB" else pil_img
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


def detect_yolo_boxes(model: "YOLO", pil_img: Image.Image, conf: float) -> List[Box]:
    det_img = pil_img.convert("RGB") if pil_img.mode != "RGB" else pil_img
    arr = np.array(det_img)
    results = model.predict(arr, verbose=False, conf=conf)
    if not results:
        return []
    r0 = results[0]
    if r0.boxes is None or len(r0.boxes) == 0:
        return []

    # COCO ids: person=0, cat=15, dog=16, horse=17, sheep=18, cow=19, elephant=20, bear=21, zebra=22, giraffe=23
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


# --- jpegtran helpers -------------------------------------------------------

def run_jpegtran(args: List[str]) -> None:
    """
    Run jpegtran, raising a helpful error if it fails.
    """
    try:
        proc = subprocess.run(args, capture_output=True, text=True)
    except FileNotFoundError:
        raise RuntimeError("jpegtran not found on PATH. Install libjpeg-turbo/mozjpeg and ensure jpegtran is available.")
    if proc.returncode != 0:
        msg = (proc.stderr or proc.stdout or "").strip()
        raise RuntimeError(f"jpegtran failed (code {proc.returncode}): {msg}")


def exif_orientation(path: Path) -> int:
    """
    Return EXIF orientation value if present, else 1.
    """
    try:
        img = Image.open(path)
        exif = img.getexif()
        # 274 = Orientation
        return int(exif.get(274, 1) or 1)
    except Exception:
        return 1


def jpegtran_normalize_orientation(in_jpg: Path, out_jpg: Path) -> None:
    """
    Losslessly normalize to orientation=1 using jpegtran transforms.
    Keeps markers via -copy all.
    """
    ori = exif_orientation(in_jpg)

    # Map EXIF orientation to jpegtran transform flags
    # 1: none
    # 2: mirror horizontal        -> -flip horizontal
    # 3: rotate 180               -> -rotate 180
    # 4: mirror vertical          -> -flip vertical
    # 5: transpose                -> -transpose
    # 6: rotate 90 CW             -> -rotate 90
    # 7: transverse               -> -transverse
    # 8: rotate 270 CW            -> -rotate 270
    transform: List[str] = []
    if ori == 2:
        transform = ["-flip", "horizontal"]
    elif ori == 3:
        transform = ["-rotate", "180"]
    elif ori == 4:
        transform = ["-flip", "vertical"]
    elif ori == 5:
        transform = ["-transpose"]
    elif ori == 6:
        transform = ["-rotate", "90"]
    elif ori == 7:
        transform = ["-transverse"]
    elif ori == 8:
        transform = ["-rotate", "270"]
    else:
        transform = []

    # -copy all keeps ICC/EXIF markers as much as jpegtran supports.
    # Note: jpegtran may keep the EXIF orientation tag; the pixels are now oriented correctly.
    # Most viewers then still show correctly. If you want to strip/clear orientation tag,
    # that's a separate EXIF-edit step (not lossless pixel-wise but metadata-wise).
    args = ["jpegtran", "-copy", "all"] + transform + ["-outfile", str(out_jpg), str(in_jpg)]
    run_jpegtran(args)


def jpegtran_lossless_crop(in_jpg: Path, out_jpg: Path, x: int, y: int, w: int, h: int) -> None:
    """
    Lossless crop using jpegtran. -trim ensures MCU-aligned crop.
    """
    crop_spec = f"{w}x{h}+{x}+{y}"
    args = ["jpegtran", "-copy", "all", "-trim", "-crop", crop_spec, "-outfile", str(out_jpg), str(in_jpg)]
    run_jpegtran(args)


# --- PNG (and fallback) saving ---------------------------------------------

def save_png_lossless(pil_img: Image.Image, out_path: Path, icc_profile: Optional[bytes]) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    save_kwargs = dict(format="PNG", compress_level=0)
    if icc_profile is not None:
        save_kwargs["icc_profile"] = icc_profile
    pil_img.save(out_path, **save_kwargs)


# --- Main per-file processing ----------------------------------------------

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
    tmp_dir: Path,
) -> Dict[str, str]:
    try:
        ext = in_path.suffix.lower()
        out_path = build_output_path(in_path, input_root, output_root, suffix, keep_subdirs)
        ensure_dir(out_path.parent)

        # Fast path: if skipping and already ratio, just copy bytes losslessly
        if skip_if_already_ratio:
            with Image.open(in_path) as img_probe:
                img_probe = ImageOps.exif_transpose(img_probe) if ext == ".png" else img_probe  # PNG orientation can be applied safely
                w0, h0 = img_probe.size
                if math.isclose((w0 / h0), target_ratio, rel_tol=1e-9, abs_tol=1e-9):
                    shutil.copy2(in_path, out_path)
                    return {"file": str(in_path), "status": "copied (already ratio)"}

        if ext in (".jpg", ".jpeg"):
            # 1) Normalize orientation losslessly into temp JPEG for consistent coordinates
            norm_path = tmp_dir / (in_path.stem + ".__norm.jpg")
            jpegtran_normalize_orientation(in_path, norm_path)

            # 2) Load normalized JPEG for detection and size (no EXIF transpose needed now)
            img = Image.open(norm_path)
            w, h = img.size

            crop_w, crop_h = compute_crop_dims_preserve_long_side(w, h, target_ratio)

            face_boxes = [b.clamp(w, h) for b in detect_faces_mtcnn(mtcnn, img)]

            body_boxes: List[Box] = []
            if (not face_only) and (yolo_model is not None):
                body_boxes = [b.clamp(w, h) for b in detect_yolo_boxes(yolo_model, img, conf=yolo_conf)]

            center = weighted_center(face_boxes + body_boxes, fallback=(w / 2.0, h / 2.0))
            x0, y0, x1, y1 = crop_rect_from_center(w, h, crop_w, crop_h, center)

            # Shift to include faces, then bodies
            x0f, y0f = adjust_to_include_union(float(x0), float(y0), crop_w, crop_h, w, h, face_boxes)
            x0b, y0b = adjust_to_include_union(x0f, y0f, crop_w, crop_h, w, h, body_boxes)

            x0 = int(round(x0b))
            y0 = int(round(y0b))
            x0 = max(0, min(x0, w - crop_w))
            y0 = max(0, min(y0, h - crop_h))

            # 3) Lossless crop with jpegtran (-trim MCU alignment)
            jpegtran_lossless_crop(norm_path, out_path, x0, y0, crop_w, crop_h)

            # Cleanup temp normalized file
            try:
                norm_path.unlink(missing_ok=True)
            except Exception:
                pass

            meta = "faces" if face_boxes else "no-faces"
            if body_boxes:
                meta += "+yolo"
            return {"file": str(in_path), "status": f"ok (jpegtran, {meta})"}

        elif ext == ".png":
            # PNG: lossless crop via Pillow (no color change)
            img = Image.open(in_path)
            img = ImageOps.exif_transpose(img)
            w, h = img.size
            icc = img.info.get("icc_profile", None)

            crop_w, crop_h = compute_crop_dims_preserve_long_side(w, h, target_ratio)

            face_boxes = [b.clamp(w, h) for b in detect_faces_mtcnn(mtcnn, img)]
            body_boxes: List[Box] = []
            if (not face_only) and (yolo_model is not None):
                body_boxes = [b.clamp(w, h) for b in detect_yolo_boxes(yolo_model, img, conf=yolo_conf)]

            center = weighted_center(face_boxes + body_boxes, fallback=(w / 2.0, h / 2.0))
            x0, y0, x1, y1 = crop_rect_from_center(w, h, crop_w, crop_h, center)

            x0f, y0f = adjust_to_include_union(float(x0), float(y0), crop_w, crop_h, w, h, face_boxes)
            x0b, y0b = adjust_to_include_union(x0f, y0f, crop_w, crop_h, w, h, body_boxes)

            x0 = int(round(x0b))
            y0 = int(round(y0b))
            x0 = max(0, min(x0, w - crop_w))
            y0 = max(0, min(y0, h - crop_h))

            cropped = img.crop((x0, y0, x0 + crop_w, y0 + crop_h))
            save_png_lossless(cropped, out_path, icc)

            meta = "faces" if face_boxes else "no-faces"
            if body_boxes:
                meta += "+yolo"
            return {"file": str(in_path), "status": f"ok (png, {meta})"}

        else:
            return {"file": str(in_path), "status": f"skipped (unsupported ext: {ext})"}

    except Exception as e:
        return {"file": str(in_path), "status": f"ERROR: {e}"}


# --- CLI -------------------------------------------------------------------

def main() -> None:
    ap = argparse.ArgumentParser(description="Smart aspect-ratio cropper (JPEG via lossless jpegtran).")
    ap.add_argument("--input", required=True, type=str, help="Input folder of images.")
    ap.add_argument("--output", required=True, type=str, help="Output folder for cropped images.")
    ap.add_argument("--ratio", required=True, type=str, help='Target aspect ratio, e.g. "1:1", "4:5", "16:9".')
    ap.add_argument("--suffix", default="", type=str, help="Optional suffix appended to filename stem.")
    ap.add_argument("--recursive", action="store_true", help="Recurse into subfolders.")
    ap.add_argument("--keep-subdirs", action="store_true", help="Preserve input subfolder structure in output.")
    ap.add_argument("--face-only", action="store_true", help="Use faces only (no YOLO bias).")
    ap.add_argument("--use-yolo", action="store_true", help="Use YOLO to bias away from clipping bodies/animals.")
    ap.add_argument("--yolo-model", default="yolov8n.pt", type=str, help="YOLO model path/name (ultralytics).")
    ap.add_argument("--yolo-conf", default=0.35, type=float, help="YOLO confidence threshold.")
    ap.add_argument("--skip-if-already-ratio", action="store_true", help="If already matches ratio, just copy.")
    ap.add_argument("--device", default="auto", choices=["auto", "cuda", "cpu"], help="Force device selection.")
    args = ap.parse_args()

    input_dir = Path(args.input).expanduser().resolve()
    output_dir = Path(args.output).expanduser().resolve()
    ensure_dir(output_dir)

    target_ratio = parse_aspect_ratio(args.ratio)

    # Torch device (for detection)
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

    tmp_dir = output_dir / ".__tmp_jpegtran"
    ensure_dir(tmp_dir)

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
            tmp_dir=tmp_dir,
        )
        results.append(res)

    # Cleanup temp dir (best-effort)
    try:
        for f in tmp_dir.glob("*"):
            f.unlink(missing_ok=True)
        tmp_dir.rmdir()
    except Exception:
        pass

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