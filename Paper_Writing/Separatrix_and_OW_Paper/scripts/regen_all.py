"""
regen_all.py  --  regenerate all (or a subset of) figures for Paper_Draft_1A.tex.

Reads figures.yaml, runs each figure script in order, and prints a summary
table when done.

Usage:
    python3 scripts/regen_all.py                  # all figures, compile at end
    python3 scripts/regen_all.py --skip-slow       # skip Monte Carlo sweeps
    python3 scripts/regen_all.py --only sep traj   # only named figures
    python3 scripts/regen_all.py --no-compile      # figures only, no pdflatex
    python3 scripts/regen_all.py --dry-run         # print what would run
"""
import sys
import os
import argparse
import subprocess
import time
from pathlib import Path

# Locate the paper dir from this file's location.
_SCRIPTS_DIR = Path(__file__).resolve().parent
PAPER_DIR    = _SCRIPTS_DIR.parent
FIGURES_DIR  = PAPER_DIR / "figures"
MANIFEST     = PAPER_DIR / "figures.yaml"
VENV_PYTHON  = (PAPER_DIR.parent.parent
                / "trunk" / "Python_Simulations"
                / "Vector_Fields" / "VF_Robot" / "venv" / "bin" / "python3")


def _python_exe():
    if VENV_PYTHON.exists():
        return str(VENV_PYTHON)
    return sys.executable


def main():
    parser = argparse.ArgumentParser(
        description="Regenerate all figures for Paper_Draft_1A.tex.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--skip-slow", action="store_true",
                        help="Skip figures marked 'slow: true' in figures.yaml.")
    parser.add_argument("--only", nargs="+", metavar="NAME",
                        help="Only regenerate these figure names.")
    parser.add_argument("--no-compile", action="store_true",
                        help="Do not recompile the paper after figures are done.")
    parser.add_argument("--dry-run", action="store_true",
                        help="Print what would be run without actually running it.")
    args = parser.parse_args()

    try:
        import yaml
    except ImportError:
        print("PyYAML not found. Run: pip install pyyaml")
        sys.exit(1)

    with open(MANIFEST) as f:
        manifest = yaml.safe_load(f)

    python = _python_exe()
    rows = []

    for name, entry in manifest.items():
        if args.only and name not in args.only:
            rows.append((name, "skipped", "--"))
            continue
        if args.skip_slow and entry.get("slow", False):
            rows.append((name, "skipped (slow)", "--"))
            continue

        script = PAPER_DIR / entry["script"]
        cmd = [python, str(script), "--no-compile"]

        if args.dry_run:
            print("would run:", " ".join(str(c) for c in cmd))
            rows.append((name, "dry-run", "--"))
            continue

        print(f"\n{'='*60}")
        print(f"  {name}")
        print(f"{'='*60}")
        t0 = time.time()
        result = subprocess.run(cmd, cwd=str(PAPER_DIR))
        elapsed = time.time() - t0
        status = "ok" if result.returncode == 0 else f"FAILED (rc={result.returncode})"
        rows.append((name, status, f"{elapsed:.1f}s"))

    # Compile once at the end (unless --no-compile).
    compile_status = "--"
    if not args.no_compile and not args.dry_run:
        from _common import compile_paper, TEX_FILE
        print(f"\n{'='*60}")
        print("  final compile")
        print(f"{'='*60}")
        rc = compile_paper(TEX_FILE)
        compile_status = "ok" if rc == 0 else f"FAILED (rc={rc})"

    # Summary table.
    print(f"\n{'='*60}")
    print(f"  Summary")
    print(f"{'='*60}")
    col_w = max(len(r[0]) for r in rows) + 2
    for name, status, elapsed in rows:
        print(f"  {name:<{col_w}}  {status:<20}  {elapsed}")
    print(f"\n  Paper compile: {compile_status}")


if __name__ == "__main__":
    sys.path.insert(0, str(_SCRIPTS_DIR := Path(__file__).resolve().parent))
    main()
