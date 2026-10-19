"""Strip notebook outputs before committing: keep the static figures (PNG) and results, drop everything else.

    uv run python scripts/strip_notebook_outputs.py notebooks/*.ipynb        # keep PNG figures
    uv run python scripts/strip_notebook_outputs.py --all notebooks/unsplash_umap.ipynb   # clear every output

Kept: PNG figures, Markdown (e.g. tables) and the plain-text value of a cell's last expression.
Dropped: printed text and logs (UMAP's progress, warnings with local paths), HTML and JavaScript (FiftyOne App,
Plotly, the interactive widgets: GitHub doesn't run them, and they weigh several MB), execution counts and widget
state. Use --all for notebooks whose figures can't be published (Unsplash photos).
"""

import argparse
import json
from pathlib import Path

KEEP = {"image/png", "text/markdown"}

parser = argparse.ArgumentParser()
parser.add_argument("notebooks", nargs="+", type=Path)
parser.add_argument("--all", action="store_true", help="clear every output, figures included")
args = parser.parse_args()

for path in args.notebooks:
    nb = json.loads(path.read_text())
    nb["metadata"].pop("widgets", None)
    for cell in nb["cells"]:
        if cell["cell_type"] != "code":
            continue
        cell["execution_count"] = None
        cell["metadata"].pop("execution", None)
        outputs = []
        for out in [] if args.all else cell["outputs"]:
            data = {k: v for k, v in out.get("data", {}).items() if k in KEEP}
            if out["output_type"] == "execute_result" and "text/html" not in out["data"]:
                data |= {k: v for k, v in out["data"].items() if k == "text/plain"}
            if data:
                outputs.append({"output_type": "display_data", "data": data, "metadata": {}})
        cell["outputs"] = outputs
    path.write_text(json.dumps(nb, indent=1, ensure_ascii=False) + "\n")
    print(f"{path}: {path.stat().st_size // 1024} KB")
