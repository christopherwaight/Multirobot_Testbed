"""Side-by-side of panel (d) from several review/ renders. Run from VF_Robot:
venv/bin/python3 experiments/ocean_small_rho/contact_sheet.py out.png name:title [name:title ...]"""
import os
import sys
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.image as mpimg

HERE = os.path.dirname(os.path.abspath(__file__))
X0, X1, Y0, Y1 = 2185, 2862, 12, 600     # panel (d) in the 2864 x 832 render

items = [a.split(":", 1) for a in sys.argv[2:]]
fig, axes = plt.subplots(1, len(items), figsize=(3.2 * len(items), 3.2))
for ax, (name, title) in zip(axes, items):
    img = mpimg.imread(os.path.join(HERE, "review", name, "ocean_progression_2km.png"))
    ax.imshow(img[Y0:Y1, X0:X1])
    ax.set_title(title, fontsize=9)
    ax.axis("off")
fig.tight_layout()
fig.savefig(os.path.join(HERE, "review", sys.argv[1]), dpi=200)
