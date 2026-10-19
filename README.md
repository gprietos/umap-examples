# umap-examples

Example code and figures used my blog post **[Dimensionality Reduction & Embedding Visualization — UMAP](https://gprietos.github.io/posts/umap/)**. This post explains how UMAP works and then shows it in action on real data. 

<!-- 
This repo has the code for every figure in the post, the two interactive ones included, plus one extra example. -->

## Notebooks

| Notebook | Section of the post | What it makes |
| --- | --- | --- |
| [`mnist_umap.ipynb`](notebooks/mnist_umap.ipynb) | [MNIST Dataset](https://gprietos.github.io/posts/umap/#mnist-dataset) | Interactive 3-D UMAP of the test digits, unsupervised vs. supervised UMAP, metric learning (kNN accuracy table), the post's cover |
| [`openimages_umap.ipynb`](notebooks/openimages_umap.ipynb) | [Open Images Dataset](https://gprietos.github.io/posts/umap/#open-images-dataset) | DINOv2 + UMAP map of 25,000 photos, without labels and colored by label group and class, with k = 50 vs. 15 |
| [`umap_explorer.ipynb`](notebooks/umap_explorer.ipynb) | [Overview](https://gprietos.github.io/posts/umap/#overview), [Effect of k and min_dist](https://gprietos.github.io/posts/umap/#effect-of-k-and-min_dist-hyperparameters) | The mammoth 3-D → 2-D figure, and the interactive explorer (3-D shapes, sliders for n_neighbors and min_dist) |
| [`unsplash_umap.ipynb`](notebooks/unsplash_umap.ipynb) | Not in the post (extra!😊) | The Open Images pipeline on Unsplash Lite photos, colored by keywords, with a 3-D map. |

<!-- The notebooks only call functions from the `umap_examples` package in [`src/umap_examples/`](src/umap_examples/), so the logic lives in plain Python files. -->

On GitHub, the interactive figures appear only as a static preview. To use them, run the notebooks or see them in the post.

## Setup

The environment is managed with uv and Python 3.12:

```bash
git clone https://github.com/gprietos/umap-examples.git
cd umap-examples
uv sync
uv run python -m ipykernel install --user --name umap-examples
```

<!-- Then open a notebook with the `umap-examples` kernel, or run one headless:

```bash
uv run jupyter nbconvert --to notebook --execute --inplace notebooks/mnist_umap.ipynb
```

Start with **MNIST**: it only downloads about 50 MB and needs no GPU. -->


## What each notebook needs

- **MNIST** and **3-D shapes** (umap_explorer): the data downloads automatically on the first run (about 50 MB). Fitting the UMAPs takes a few minutes for MNIST and about 25 minutes for the explorer (to compute it faster, [`scripts/`](scripts/)). The results are cached, so later runs are quick.
- **Open Images**: the first run downloads about 7 GB of images and embeds them, which takes a few minutes on a GPU and much longer on a CPU. The results are stored, so later runs are quick.
- **Unsplash**: download Unsplash Lite from [unsplash.com/data](https://unsplash.com/data), put or symlink the extracted folder at `data/unsplash/`, then download the images (about 2.5 GB) with:

  ```bash
  uv run python scripts/download_unsplash.py
  ```

All data goes under `data/` and figures under `outputs/figures/`. The default paths are in [`config.py`](src/umap_examples/config.py).


## Where results are stored

- **FiftyOne datasets** (`mnist`, `open-images-v7-validation-25k`, `unsplash-lite`) are persistent and live in FiftyOne's database under `~/.fiftyone`. Embeddings are stored in fields `emb_<model>`, and UMAP layouts as brain runs, which the FiftyOne App shows in its Embeddings panel next to the images. Re-running a notebook only loads them.
- **Other UMAPs** (MNIST pixels, supervised UMAP, the 3-D shapes) are saved in `data/cache/`, along with the fitted models. Deleting the folder is safe: they are recomputed on the next run.

## Repository layout

| Folder | What it has |
| --- | --- |
| [`notebooks/`](notebooks/) | the four notebooks |
| [`src/umap_examples/`](src/umap_examples/) | the package the notebooks use (data loading, embeddings, UMAP, plotting) |
| [`scripts/`](scripts/) | data downloads and exports, and stripping notebook outputs |
| [`blog/`](blog/) | the interactive figures: how they work, generating their data, and trying them outside the blog |


<!-- ## Notebook outputs

The notebooks are published with their figures but without logs, HTML or JavaScript outputs, and the Unsplash notebook without any outputs. They were stripped with [`scripts/strip_notebook_outputs.py`](scripts/strip_notebook_outputs.py). -->

## Data and licenses

The code is under the [MIT License](LICENSE). The data and models have their own licenses:

- **MNIST**: Y. LeCun, C. Cortes and C. J. C. Burges, [The MNIST Database](https://yann.lecun.com/exdb/mnist/), CC BY-SA 3.0.
- **Open Images V7**: [Google](https://storage.googleapis.com/openimages/web/index.html). The annotations are CC BY 4.0, and the images are listed as CC BY 2.0, so they need attribution if you share them.
- **Unsplash Lite**: [Unsplash](https://unsplash.com/data), under the Unsplash Lite terms. They allow downloading and internal use, not sharing the photos or other parts of the dataset, so the Unsplash notebook is published without outputs.
- **Mammoth**: a scan from the Smithsonian, as prepared by PAIR for [Understanding UMAP](https://github.com/PAIR-code/understanding-umap) (Apache-2.0). It is downloaded from that repository on first use.
- **DINOv2** weights: Meta, Apache-2.0, via [timm](https://github.com/huggingface/pytorch-image-models).
