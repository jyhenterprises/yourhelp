# Requirements
# choco install -y imagemagick
# uv pip install pillow pillow-heif piexif
# USAGE
# python heic_to_jpg_magick.py "./heic" -o "./heic_to_jpeg" --mirror-tree -q 95

import argparse
import subprocess
from pathlib import Path

def convert_heic_to_jpg(src: Path, dst: Path, quality: int = 95) -> None:
    dst.parent.mkdir(parents=True, exist_ok=True)

    # ImageMagick notes:
    # - auto-orient: respects EXIF orientation so portraits aren’t sideways
    # - quality: JPEG quality (95 is typically visually lossless)
    # - strip is NOT used so metadata is generally preserved (IM usually keeps EXIF unless stripped)
    cmd = [
        "magick",
        str(src),
        "-auto-orient",
        "-quality", str(quality),
        str(dst),
    ]

    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode != 0:
        raise RuntimeError(
            f"ImageMagick failed for {src}\nSTDOUT: {result.stdout}\nSTDERR: {result.stderr}"
        )

def iter_heic_files(input_path: Path):
    if input_path.is_file():
        if input_path.suffix.lower() == ".heic":
            yield input_path
        return
    yield from (p for p in input_path.rglob("*") if p.is_file() and p.suffix.lower() == ".heic")

def main():
    p = argparse.ArgumentParser(description="Batch convert .HEIC photos to .JPG using ImageMagick (installed via Chocolatey).")
    p.add_argument("input", help="Input file or folder")
    p.add_argument("-o", "--output", help="Output folder (default: <input>/jpg_out for folders, or alongside file)")
    p.add_argument("-q", "--quality", type=int, default=95, help="JPEG quality 1-100 (default 95)")
    p.add_argument("--mirror-tree", action="store_true", help="Mirror folder structure under output when input is a folder")
    args = p.parse_args()

    input_path = Path(args.input).expanduser().resolve()
    if not input_path.exists():
        raise SystemExit(f"Input path not found: {input_path}")

    if args.output:
        output_root = Path(args.output).expanduser().resolve()
    else:
        output_root = (input_path / "jpg_out") if input_path.is_dir() else input_path.parent

    total = 0
    ok = 0

    for src in iter_heic_files(input_path):
        total += 1
        if input_path.is_dir() and args.mirror_tree:
            rel = src.relative_to(input_path)
            dst = (output_root / rel).with_suffix(".jpg")
        else:
            dst = (output_root / src.name).with_suffix(".jpg")

        try:
            convert_heic_to_jpg(src, dst, quality=args.quality)
            ok += 1
            print(f"OK:  {src} -> {dst}")
        except Exception as e:
            print(f"FAIL:{src}\n  {e}")

    print(f"\nDone. Converted {ok}/{total}. Output: {output_root}")

if __name__ == "__main__":
    main()