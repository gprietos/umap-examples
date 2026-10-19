"""Unsplash Lite dataset: metadata tables, image download, FiftyOne dataset."""

import glob
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import pandas as pd
import requests
from tqdm import tqdm

from .config import UNSPLASH_DIR


### ------------------------------------------------
### --- Metadata -----------------------------------
### ------------------------------------------------

def load_table(name, dataset_dir=UNSPLASH_DIR):
    """Load one table (e.g. "photos", "keywords"), concatenating its .tsv000, .tsv001, ... parts.

    Same approach as https://github.com/unsplash/datasets/blob/master/how-to/python/pandas.ipynb
    """
    files = sorted(glob.glob(str(Path(dataset_dir) / f"{name}.tsv*")))
    if not files:
        raise FileNotFoundError(f"No {name}.tsv* files in {dataset_dir}")
    return pd.concat(
        (pd.read_csv(f, sep="\t", header=0, low_memory=False) for f in files),
        axis=0, ignore_index=True,
    )


def load_tables(dataset_dir=UNSPLASH_DIR):
    """Return the photos and keywords tables."""
    return load_table("photos", dataset_dir), load_table("keywords", dataset_dir)


def keywords_per_photo(keywords, min_confidence=50.0):
    """Map photo_id -> list of keywords.

    Keeps keywords set by a user, or given by one of the AI services with at least `min_confidence`.
    """
    conf = keywords[["ai_service_1_confidence", "ai_service_2_confidence"]].max(axis=1)
    keep = (keywords["suggested_by_user"] == "t") | (conf >= min_confidence)
    kept = keywords[keep].dropna(subset=["keyword"])
    return kept.groupby("photo_id")["keyword"].apply(lambda s: sorted(set(s))).to_dict()


### ------------------------------------------------
### --- Image download -----------------------------
### ------------------------------------------------

def image_path(photo_id, out_dir=UNSPLASH_DIR / "images"):
    return Path(out_dir) / f"{photo_id}.jpg"


def _download_one(photo_id, url, out_dir, width, retries):
    path = image_path(photo_id, out_dir)
    if path.exists():
        return photo_id, True
    # photo_image_url is a dynamic imgix URL, so the image is resized on Unsplash's side
    params = {"w": width, "fm": "jpg", "q": 80}
    for attempt in range(retries):
        try:
            r = requests.get(url, params=params, timeout=30)
            r.raise_for_status()
            tmp = path.with_suffix(".part")
            tmp.write_bytes(r.content)
            tmp.rename(path)
            return photo_id, True
        except requests.RequestException:
            time.sleep(2 ** attempt)
    return photo_id, False


def download_images(photos, out_dir=UNSPLASH_DIR / "images", width=640, workers=16, retries=3):
    """Download every photo in `photos` to out_dir/<photo_id>.jpg. Existing files are skipped.

    Returns the list of photo_ids that failed.
    """
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    failed = []
    with ThreadPoolExecutor(max_workers=workers) as pool:
        futures = [
            pool.submit(_download_one, pid, url, out_dir, width, retries)
            for pid, url in zip(photos["photo_id"], photos["photo_image_url"])
        ]
        for fut in tqdm(as_completed(futures), total=len(futures), desc="Downloading"):
            photo_id, ok = fut.result()
            if not ok:
                failed.append(photo_id)
    return failed


### ------------------------------------------------
### --- FiftyOne dataset ---------------------------
### ------------------------------------------------

def load_dataset(name="unsplash-lite", dataset_dir=UNSPLASH_DIR, images_dir=None):
    """Load the persistent FiftyOne dataset, creating it if needed, with one sample per downloaded image.

    Only images not yet in the dataset are added, so existing fields (embeddings) and brain runs (UMAPs) are kept.
    `images_dir` defaults to `dataset_dir/images`.
    """
    import fiftyone as fo

    images_dir = Path(images_dir or Path(dataset_dir) / "images")
    if fo.dataset_exists(name):
        dataset = fo.load_dataset(name)
    else:
        dataset = fo.Dataset(name, persistent=True)

    photos, keywords = load_tables(dataset_dir)
    known = set(dataset.distinct("photo_id")) if len(dataset) else set()
    new = photos[[pid not in known and image_path(pid, images_dir).exists() for pid in photos["photo_id"]]]
    if new.empty:
        return dataset

    kw = keywords_per_photo(keywords)

    def text(v):
        return None if pd.isna(v) else str(v)

    dataset.add_samples([
        fo.Sample(
            filepath=str(image_path(row.photo_id, images_dir)),
            photo_id=row.photo_id,
            description=text(row.photo_description),
            ai_description=text(row.ai_description),
            keywords=kw.get(row.photo_id, []),
            country=text(row.photo_location_country),
            photographer=text(row.photographer_username),
        )
        for row in new.itertuples()
    ])
    return dataset


def keyword_labels(keywords_per_sample, targets, other="other"):
    """One label per sample: the first of `targets` found in its keywords, else `other`.

    Keywords are independent of the embeddings, so coloring by them shows what the embedding groups together.
    """
    labels = []
    for kws in keywords_per_sample:
        kws = set(kws or [])
        labels.append(next((t for t in targets if t in kws), other))
    return labels
