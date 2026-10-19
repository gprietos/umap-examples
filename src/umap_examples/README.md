# umap_examples


| Module | What it has |
| --- | --- |
| [`config.py`](config.py) | default paths (all under `data/`) and the seed (51) |
| [`mnist.py`](mnist.py) | MNIST pixels, its FiftyOne dataset, kNN, data for the 3-D MNIST figure |
| [`openimages.py`](openimages.py) | Open Images sample (FiftyOne zoo) and labels from the class hierarchy |
| [`unsplash.py`](unsplash.py) | Unsplash Lite metadata, image download, FiftyOne dataset, keywords |
| [`embeddings.py`](embeddings.py) | image models (DINOv2 via timm, CLIP via the FiftyOne zoo) and embedding a dataset |
| [`reduction.py`](reduction.py) | UMAP of a FiftyOne field (brain runs) or of an array (cached with joblib) |
| [`shapes.py`](shapes.py) | 3-D toy shapes: mammoth, swiss roll, linked rings, torus, blobs |
| [`umap_explorer.py`](umap_explorer.py) | data for the UMAP explorer figure |
| [`widgets.py`](widgets.py) | the interactive figures in a notebook |
| [`plotting.py`](plotting.py) | the static figures (also saved in a dark version for the blog) |

Every function that reads or writes files takes its path as an argument; the defaults come from `config.py`.
