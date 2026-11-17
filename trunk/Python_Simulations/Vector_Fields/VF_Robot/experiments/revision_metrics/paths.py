"""Output folders for the revision_metrics scripts.

Set REVISION_OUT to move all outputs, e.g. to <repo>/analysis/simulation/revision_metrics.
Unset, they stay next to the scripts (results/, tables/, figures/).
"""
import os

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.environ.get("REVISION_OUT", HERE)
RES = os.path.join(ROOT, "results")
TABLES = os.path.join(ROOT, "tables")
FIGURES = os.path.join(ROOT, "figures")
for _d in (RES, TABLES, FIGURES):
    os.makedirs(_d, exist_ok=True)
