# Blog widgets

The post's two interactive figures, as plain JavaScript that draws UMAP layouts precomputed in Python.

- **3-D MNIST**: every test digit drawn as its own image with [scatter-gl](https://github.com/PAIR-code/scatter-gl), PAIR's 3-D scatter plot library.
- **UMAP explorer**: a 3-D shape next to its 2-D UMAP, drawn with Plotly, for a grid of n_neighbors × min_dist.

## Layout

```
_includes/umap/          the HTML snippets the post includes (mnist_3d.html, umap_explorer.html)
assets/js/umap/          the widgets' code
assets/css/umap/         their styles
assets/data/umap/        the precomputed layouts (generate them first, see below)
umap_explorer_standalone.html   the explorer on a page of its own (to test it, and the source for the notebooks)
mnist_3d_standalone.html        the 3-D MNIST on a page of its own
```

## Generating the data

The widgets read their data from `assets/data/`. Generate it with the export scripts (see [`scripts/`](../scripts/)):

```bash
uv run python scripts/export_mnist_3d.py
uv run python scripts/export_umap_explorer.py
```

## Previewing

To try the widgets out, serve this folder and open a standalone page:

```bash
cd blog && python -m http.server   # then open localhost:8000/umap_explorer_standalone.html or /mnist_3d_standalone.html (add ?dark for dark mode)
```

In the notebooks, [`widgets.py`](../src/umap_examples/widgets.py) packs a standalone page and all its files (code, styles and data) into a single HTML document and shows it in a frame inside the notebook. The drawing libraries (Plotly, scatter-gl) are downloaded from the internet each time, so the widgets need a connection.
