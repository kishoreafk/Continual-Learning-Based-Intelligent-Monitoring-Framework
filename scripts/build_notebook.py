"""Assemble notebooks/cells/*.{md,txt} into a single Colab notebook.

Run:  python scripts/build_notebook.py
"""

import base64
import json
from pathlib import Path

ROOT = Path("E:/Continual Monitoring Framework")
PKG = ROOT / "sage_monitor"
CELLS = ROOT / "notebooks" / "cells"
OUT = ROOT / "notebooks" / "SAGE_Monitor_Colab_Training.ipynb"

ORDER = [
    "00_header.md",
    "01_install.txt",
    "02_data.md",
    "03_download.txt",
    "03b_inspect.txt",
    "03c_setup_pkg.txt",
    "04_load.txt",
    "05_build.txt",
    "06_stream.txt",
    "07_train.txt",
    "08_eval.txt",
    "09_plot.txt",
    "10_save.txt",
]


def main():
    # embed the package source keyed by basename
    files = {}
    for p in sorted(PKG.glob("*.py")):
        files[p.name] = base64.b64encode(p.read_bytes()).decode("ascii")
    pkg_json = json.dumps(files)

    cells = []
    for name in ORDER:
        text = (CELLS / name).read_text(encoding="utf-8")
        if name == "03c_setup_pkg.txt":
            text = text.replace("__PKG_FILES__", pkg_json)
        kind = "markdown" if name.endswith(".md") else "code"
        cell = {
            "cell_type": kind,
            "metadata": {},
            "source": text.splitlines(keepends=True),
        }
        if kind == "code":
            cell["execution_count"] = None
            cell["outputs"] = []
        cells.append(cell)

    nb = {
        "cells": cells,
        "metadata": {
            "kernelspec": {"display_name": "Python 3", "name": "python3"},
            "language_info": {"name": "python"},
            "accelerator": "GPU",
        },
        "nbformat": 4,
        "nbformat_minor": 5,
    }
    OUT.write_text(json.dumps(nb, indent=1), encoding="utf-8")
    print(f"wrote {OUT} ({OUT.stat().st_size / 1024:.1f} KB, {len(cells)} cells)")


if __name__ == "__main__":
    main()
