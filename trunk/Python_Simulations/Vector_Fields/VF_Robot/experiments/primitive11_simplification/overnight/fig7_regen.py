"""Regenerate Fig. 7 (noise_seven_starts_col.png) from a chosen CSV set, into a
review folder instead of the paper's figures/. Usage: fig7_regen.py <suffix> <out_dir>
e.g. fig7_regen.py final overnight/figures_for_review  -> reads noise_both_S1_final.csv ..."""
import os, sys
here = os.path.dirname(os.path.abspath(__file__))
root = os.path.dirname(os.path.dirname(os.path.dirname(here)))
sys.path.insert(0, root); os.chdir(root)
import matplotlib; matplotlib.use("Agg")
import experiments.plot_flip_resolution_rows as R
suffix, out = sys.argv[1], os.path.abspath(sys.argv[2])
os.makedirs(out, exist_ok=True)
R.FIG_DIR = out
R.CSV_FILES.clear(); R.CSV_FILES.update({f"S{i}": f"noise_both_S{i}_{suffix}.csv" for i in range(1, 8)})
sys.argv = ["plot_flip_resolution_rows.py", "--paper", "--column", "--out-dir", out]
R.main()
print("written to", out)
