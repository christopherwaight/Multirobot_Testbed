"""
3-robot vector plane-fit primitives on one configurable field.

Runs critical_point_plane_fitting (attraction) from six starts and
critical_point_orbiter_plane_fitting at several desired radii. Fields, cluster
setup and metrics live in _field_tests_common.py (read its docstring for the
field definitions). Replaces single_gyre_plane_fit.py (2026-10-28).

Output: experiments/outputs/plane_fit_field_test/<field>_side.._a.._st../
        attract.png, orbit.png, summary.txt

Run from VF_Robot (defaults reproduce the original single-gyre run):
    venv/bin/python3 experiments/plane_fit_field_test.py
    venv/bin/python3 experiments/plane_fit_field_test.py --stiction 0
    venv/bin/python3 experiments/plane_fit_field_test.py --field potential_saddle --eps -1 --side 0.1 --alpha 0.01
"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import _field_tests_common as ft

if __name__ == "__main__":
    args = ft.build_parser(__doc__.strip().splitlines()[0],
                           os.path.join(ft.PROJECT_ROOT, "experiments", "outputs", "plane_fit_field_test")
                           ).parse_args()
    ft.run_suite("vector", args)
