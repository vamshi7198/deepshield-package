"""Builds a Kaggle notebook from a `# %%`-cell Python script.

Usage: python build_notebook.py [--smoke | --full] [--src deepshield_round2.py] [--kernel kaggle_kernel_round2]
Defaults: --src deepshield_verification.py --kernel kaggle_kernel
Writes <src>.ipynb next to this file and into <kernel>/ (the folder pushed with `kaggle kernels push`).
"""
import json, re, sys, os

HERE = os.path.dirname(os.path.abspath(__file__))


def arg(name, default):
    return sys.argv[sys.argv.index(name) + 1] if name in sys.argv else default


src_name = arg("--src", "deepshield_verification.py")
kernel_dir = arg("--kernel", "kaggle_kernel")
src = open(os.path.join(HERE, src_name), encoding="utf-8").read()
if "--full" in sys.argv:
    src = re.sub(r"^SMOKE_TEST = True", "SMOKE_TEST = False", src, count=1, flags=re.M)
elif "--smoke" in sys.argv:
    src = re.sub(r"^SMOKE_TEST = False", "SMOKE_TEST = True", src, count=1, flags=re.M)

parts = re.split(r"^# %%(.*)$", src, flags=re.M)
cells = []
for i in range(1, len(parts), 2):
    kind, body = parts[i].strip(), parts[i + 1].strip("\n")
    if kind == "[markdown]":
        body = "\n".join(l[2:] if l.startswith("# ") else l.lstrip("#") for l in body.split("\n"))
        cells.append({"cell_type": "markdown", "metadata": {}, "source": body.splitlines(True)})
    else:
        cells.append({"cell_type": "code", "metadata": {}, "execution_count": None, "outputs": [],
                      "source": body.splitlines(True)})
nb = {"cells": cells, "nbformat": 4, "nbformat_minor": 4,
      "metadata": {"kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
                   "language_info": {"name": "python"}}}
nb_name = os.path.splitext(src_name)[0] + ".ipynb"
os.makedirs(os.path.join(HERE, kernel_dir), exist_ok=True)
for out in (os.path.join(HERE, nb_name), os.path.join(HERE, kernel_dir, nb_name)):
    json.dump(nb, open(out, "w", encoding="utf-8"), indent=1, ensure_ascii=False)
mode = re.search(r"^SMOKE_TEST = \w+", src, re.M).group(0)
print(f"{nb_name}: {len(cells)} cells, {mode} -> {kernel_dir}/")
