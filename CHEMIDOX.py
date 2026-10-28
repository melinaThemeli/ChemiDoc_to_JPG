#!/usr/bin/env python3
"""
Μετατροπή αρχείων Bio-Rad .1sc (ChemiDoc XRS) σε εικόνες.

Για κάθε αρχείο παράγει:
  <όνομα>_16bit.tif     ακατέργαστα δεδομένα 16-bit (για ποσοτικοποίηση, π.χ. ImageJ)
  <όνομα>_auto.png      8-bit, αυτόματη αντίθεση, φωτεινές ζώνες σε σκούρο φόντο
  <όνομα>_inverted.png  8-bit, αντεστραμμένοι τόνοι (σκούρες ζώνες σε λευκό)

Χρήση:
  python convert_1sc.py αρχείο1.1sc αρχείο2.1sc
  python convert_1sc.py φάκελος/                  # όλα τα .1sc του φακέλου
  python convert_1sc.py αρχείο.1sc --flipv        # κατακόρυφη αναστροφή
  python convert_1sc.py αρχείο.1sc --fliph        # οριζόντια αναστροφή
  python convert_1sc.py αρχείο.1sc -o έξοδος/     # φάκελος εξόδου

Απαιτεί:  pip install numpy pillow
"""
import argparse
import struct
import sys
from pathlib import Path

import numpy as np
from PIL import Image

WIDTH, HEIGHT = 696, 520          # ChemiDoc XRS, binning 2x2
IMG_BYTES = WIDTH * HEIGHT * 2    # 16-bit


def read_1sc(path):
    data = Path(path).read_bytes()
    if b"Bio-Rad Scan File" not in data[:128]:
        raise ValueError("δεν φαίνεται να είναι αρχείο Bio-Rad .1sc")
    # Ο πίνακας περιεχομένων στην αρχή έχει ζεύγη (offset, μέγεθος).
    # Η εικόνα είναι το τμήμα που φτάνει ακριβώς μέχρι το τέλος του αρχείου.
    offset = None
    for p in range(0x80, 0x400, 4):
        o, s = struct.unpack("<II", data[p:p + 8])
        if s == IMG_BYTES and o + s == len(data):
            offset = o
            break
    if offset is None:                      # εφεδρικά: τα τελευταία bytes
        offset = len(data) - IMG_BYTES
    raw = data[offset:offset + IMG_BYTES]
    if len(raw) != IMG_BYTES:
        raise ValueError("απροσδόκητο μέγεθος δεδομένων εικόνας")
    return np.frombuffer(raw, dtype="<u2").reshape(HEIGHT, WIDTH)


def convert(path, outdir, flipv=False, fliph=False):
    img = read_1sc(path)
    if flipv:
        img = img[::-1, :]
    if fliph:
        img = img[:, ::-1]
    img = np.ascontiguousarray(img)

    stem = Path(path).stem
    Image.fromarray(img).save(outdir / f"{stem}_16bit.tif")

    lo, hi = np.percentile(img, [0.5, 99.5])
    scaled = np.clip((img.astype(float) - lo) / (hi - lo), 0, 1)
    img8 = (scaled * 255).astype("uint8")
    Image.fromarray(img8).save(outdir / f"{stem}_auto.png")
    Image.fromarray(255 - img8).save(outdir / f"{stem}_inverted.png")
    print(f"OK  {path}")


def main():
    ap = argparse.ArgumentParser(description="Μετατροπή .1sc σε TIFF/PNG")
    ap.add_argument("inputs", nargs="+", help="αρχεία .1sc ή φάκελοι")
    ap.add_argument("-o", "--outdir", default=None, help="φάκελος εξόδου")
    ap.add_argument("--flipv", action="store_true", help="κατακόρυφη αναστροφή")
    ap.add_argument("--fliph", action="store_true", help="οριζόντια αναστροφή")
    args = ap.parse_args()

    files = []
    for item in args.inputs:
        p = Path(item)
        files += sorted(p.glob("*.1sc")) if p.is_dir() else [p]

    for f in files:
        outdir = Path(args.outdir) if args.outdir else f.parent
        outdir.mkdir(parents=True, exist_ok=True)
        try:
            convert(f, outdir, args.flipv, args.fliph)
        except Exception as e:
            print(f"ΣΦΑΛΜΑ {f}: {e}", file=sys.stderr)


if __name__ == "__main__":
    main()

