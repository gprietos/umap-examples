"""Download the Unsplash Lite images (default: data/unsplash/images/). Resumable: existing files are skipped.

    uv run python scripts/download_unsplash.py [--dataset-dir DIR] [--out-dir DIR] [--width 640] [--workers 16] [--limit N]
"""

import argparse
from pathlib import Path

from umap_examples.config import UNSPLASH_DIR
from umap_examples.unsplash import download_images, load_table

parser = argparse.ArgumentParser()
parser.add_argument("--dataset-dir", type=Path, default=UNSPLASH_DIR, help=f"folder with photos.tsv000 (default: {UNSPLASH_DIR})")
parser.add_argument("--out-dir", type=Path, default=None, help="where to save the images (default: <dataset-dir>/images)")
parser.add_argument("--width", type=int, default=640)
parser.add_argument("--workers", type=int, default=16)
parser.add_argument("--limit", type=int, default=None, help="only the first N photos")
args = parser.parse_args()

photos = load_table("photos", args.dataset_dir)
if args.limit:
    photos = photos.head(args.limit)

out_dir = args.out_dir or args.dataset_dir / "images"
failed = download_images(photos, out_dir=out_dir, width=args.width, workers=args.workers)
print(f"{len(photos) - len(failed)}/{len(photos)} images available, {len(failed)} failed")
if failed:
    print("Failed:", " ".join(failed[:50]), "..." if len(failed) > 50 else "")
