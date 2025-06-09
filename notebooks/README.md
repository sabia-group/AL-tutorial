# Convert a notebook to a python script and viceversa

How to convert the notebook to a python script (after `pip install jupytext`):
```bash
jupytext tutorial.ipynb --to py && \
mv tutorial.py script.py
```

This command will create the `script.py` which is more easily "handled" by git.

In case you have some conflicts during merging, try to solve the conflicts in `script.py` rather than in `tutorial.ipynb`.

Once done, you can revert the conversion and generate a notebook with:
```bash
jupytext script.py --to notebook && \
mv script.ipynb tutorial.ipynb
```

TLDR: Long live python scripts, down with notebooks!
