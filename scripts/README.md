# Scripts


| Script | What it does | Main options |
| --- | --- | --- |
| [`download_unsplash.py`](download_unsplash.py) | Downloads the Unsplash Lite images into `data/unsplash/images/` (about 2.5 GB, resumable) | `--limit N` for a quick test, `--width` (default 640) |
| [`export_mnist_3d.py`](export_mnist_3d.py) | Writes the data for the 3-D MNIST figure into `blog/assets/data/umap/mnist_3d/` | `--out-dir` |
| [`export_umap_explorer.py`](export_umap_explorer.py) | Writes the data for the UMAP explorer into `blog/assets/data/umap/umap_explorer/` | `--shapes` to export only some shapes |
| [`strip_notebook_outputs.py`](strip_notebook_outputs.py) | Strips notebook outputs: keeps the PNG figures and results, drops logs, HTML and JavaScript | `--all` to clear every output |

The explorer export runs UMAP single-threaded (for the fixed seed) over a grid of 210 runs, so it takes about 25 minutes. To speed it up, run one process per shape:

```bash
for s in mammoth swiss_roll linked_rings torus blobs; do uv run python scripts/export_umap_explorer.py --shapes $s & done; wait
uv run python scripts/export_umap_explorer.py   # writes index.json with every shape; only loads the cached runs
```

The notebooks in this repository were stripped with:

```bash
uv run python scripts/strip_notebook_outputs.py notebooks/*.ipynb
uv run python scripts/strip_notebook_outputs.py --all notebooks/unsplash_umap.ipynb
```
