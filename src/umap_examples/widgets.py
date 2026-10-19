"""The post's two interactive figures, shown in a notebook.

The figures are the blog's own widgets (`blog/assets/js/umap/`), not copies. A widget's standalone page
(`blog/*_standalone.html`) becomes a single self-contained HTML document: its CSS and JS are inlined, and the data files,
which the blog serves next to the page, are passed as data URIs in `window.UMAP_FILES`. The document is shown in an
iframe, so the widget's styles and scripts don't leak into the notebook. Only the libraries come from a CDN, so the
figures need an internet connection.

Each figure also carries a static PNG of the same data, drawn with matplotlib. Jupyter and VS Code show the
interactive version; `scripts/strip_notebook_outputs.py` keeps only the PNG, so the committed notebooks (and GitHub,
which doesn't run JavaScript) show the image.
"""

import base64
import html
import io
import json
import re
from pathlib import Path

import numpy as np

from .config import EXPLORER_DIR, MNIST3D_DIR, REPO_DIR

BLOG_DIR = REPO_DIR / "blog"
# Inside the notebook's iframe, grow the iframe to the page's height (the iframe is same-origin, srcdoc)
FIT_HEIGHT = """<script>
if (window.frameElement) new ResizeObserver(() => {
  window.frameElement.style.height = document.documentElement.scrollHeight + "px";
}).observe(document.documentElement);
</script>
"""
MIME = {".json": "application/json", ".png": "image/png", ".css": "text/css", ".js": "text/javascript"}


def _data_uri(path):
    return f"data:{MIME[path.suffix]};base64,{base64.b64encode(path.read_bytes()).decode()}"


def build_html(page, data_dir, dark=False):
    """The widget of `blog/<page>` as one HTML document, with the files of `data_dir` inlined."""
    data_dir = Path(data_dir)
    files = {p.relative_to(data_dir).as_posix(): _data_uri(p) for p in sorted(data_dir.rglob("*")) if p.is_file()}
    doc = (BLOG_DIR / page).read_text()

    doc = re.sub(r"<!--.*?-->\n?", "", doc, flags=re.S)
    doc = re.sub(r"\s*<figcaption>.*?</figcaption>", "", doc)  # the page's placeholder caption
    # The page's ?dark switch doesn't work in an iframe: set the class directly
    doc = re.sub(r"<script>document\.addEventListener.*?</script>\n?", "", doc)
    if dark:
        doc = doc.replace("<body>", '<body class="dark">')
    doc = re.sub(r'<link rel="stylesheet" href="(assets/[^"]+)">',
                 lambda m: f"<style>{(BLOG_DIR / m[1]).read_text()}</style>", doc)
    # As data URIs, the local scripts stay `defer` and run after the CDN libraries, as on the blog
    doc = re.sub(r'<script defer src="(assets/[^"]+)"></script>',
                 lambda m: f'<script defer src="{_data_uri(BLOG_DIR / m[1])}"></script>', doc)
    doc = doc.replace("</head>", f"<script>window.UMAP_FILES = {json.dumps(files)};</script>\n{FIT_HEIGHT}</head>", 1)
    # The pages' 20px top/bottom margin is for a browser tab; in a notebook it is wasted space
    doc = doc.replace("margin: 20px auto;", "margin: 0 auto;")
    return doc


class _Widget:
    """Displayed by Jupyter as its HTML, with a static PNG in the same output for viewers without JavaScript.

    (IPython.display.HTML would warn about iframes.)
    """

    def __init__(self, html, png=None):
        self.html = html
        self.png = png

    def _repr_mimebundle_(self, include=None, exclude=None):
        bundle = {"text/html": self.html}
        if self.png is not None:
            bundle["image/png"] = base64.b64encode(self.png).decode()
        return bundle


def _iframe(doc, height, png=None):
    return _Widget(f'<iframe srcdoc="{html.escape(doc, quote=True)}" '
                   f'style="width: 100%; height: {height}px; border: 0;"></iframe>', png)


### ------------------------------------------------
### --- Static previews ----------------------------
### ------------------------------------------------

NOTE = "Static preview. Run the notebook for the interactive figure or see it in the post."


def _png(fig):
    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=100, bbox_inches="tight", facecolor="white")
    return buf.getvalue()


def _no_axes_3d(ax, X, zoom=1.0):
    ax.set_box_aspect(np.ptp(X, axis=0), zoom=zoom)
    ax.set_axis_off()


def preview_umap_explorer(data_dir=EXPLORER_DIR, shape="mammoth", ks=(5, 15, 100), min_dist="0.1"):
    """PNG of the explorer's data: the 3-D shape next to its 2-D UMAP for a few values of n_neighbors."""
    from matplotlib.figure import Figure

    data = json.loads((Path(data_dir) / f"{shape}.json").read_text())
    X = np.reshape(data["points3d"], (-1, 3))
    colors = data["colors"]

    fig = Figure(figsize=(4 * (len(ks) + 1), 4.4), layout="constrained")
    ax = fig.add_subplot(1, len(ks) + 1, 1, projection="3d")
    ax.scatter(*X.T, c=colors, s=2, linewidths=0, depthshade=False)
    ax.view_init(elev=8, azim=-10)
    _no_axes_3d(ax, X)
    ax.set_title("3-D data")
    for i, k in enumerate(ks, start=2):
        P = np.reshape(data["layouts"][f"k{k}_md{min_dist}"], (-1, 2)) / 999 * 2 - 1
        ax = fig.add_subplot(1, len(ks) + 1, i)
        ax.scatter(*P.T, c=colors, s=2, linewidths=0)
        ax.set(xticks=[], yticks=[], aspect="equal", adjustable="datalim", box_aspect=1,
               title=f"UMAP, n_neighbors={k}, min_dist={min_dist}")
    fig.supxlabel(NOTE, fontsize=10, color="0.4")
    return _png(fig)


def preview_mnist_3d(data_dir=MNIST3D_DIR):
    """PNG of the 3-D MNIST data: the test digits as points, colored by digit."""
    from matplotlib import colormaps
    from matplotlib.figure import Figure
    from matplotlib.lines import Line2D

    data = json.loads((Path(data_dir) / "mnist_3d.json").read_text())
    X, labels = np.reshape(data["points"], (-1, 3)), np.asarray(data["labels"])
    palette = colormaps["tab10"].colors

    fig = Figure(figsize=(8, 6), layout="constrained")
    ax = fig.add_subplot(projection="3d")
    ax.scatter(*X.T, c=[palette[d] for d in labels], s=2, linewidths=0)
    _no_axes_3d(ax, X, zoom=1.35)
    handles = [Line2D([], [], marker="o", ls="", color=palette[d], label=str(d)) for d in range(10)]
    ax.legend(handles=handles, title="Digit", loc="center left", bbox_to_anchor=(1, 0.5), frameon=False)
    fig.suptitle("3-D UMAP of the 10,000 MNIST test digits")
    fig.supxlabel(NOTE, fontsize=9, color="0.4")
    return _png(fig)


def _check(data_dir, files, command):
    missing = [f for f in files if not (Path(data_dir) / f).exists()]
    if missing:
        raise FileNotFoundError(f"{', '.join(missing)} not in {data_dir}. Create the data with {command}.")


def show_mnist_3d(data_dir=MNIST3D_DIR, dark=False, height=640):
    """The post's 3-D MNIST figure: the 10k test digits drawn as images at their 3-D UMAP positions.

    Drag to rotate, scroll to zoom, click a digit below the plot to highlight it (shift-click to add more).
    """
    _check(data_dir, ["mnist_3d.json", "sprites.png"], "`export_mnist_3d()` (umap_examples.mnist)")
    return _iframe(build_html("mnist_3d_standalone.html", data_dir, dark), height, preview_mnist_3d(data_dir))


def show_umap_explorer(data_dir=EXPLORER_DIR, dark=False, height=620):
    """The post's UMAP explorer: a 3-D shape and its 2-D UMAP, with sliders for n_neighbors and min_dist."""
    _check(data_dir, ["index.json"], "`export_widget_data()` (umap_examples.umap_explorer)")
    return _iframe(build_html("umap_explorer_standalone.html", data_dir, dark), height, preview_umap_explorer(data_dir))


def save_html(path, widget="mnist_3d", dark=False):
    """Write a widget ("mnist_3d" or "umap_explorer") as a standalone HTML file, to open in a browser."""
    page, data_dir = {"mnist_3d": ("mnist_3d_standalone.html", MNIST3D_DIR),
                      "umap_explorer": ("umap_explorer_standalone.html", EXPLORER_DIR)}[widget]
    Path(path).write_text(build_html(page, data_dir, dark))
    return path
