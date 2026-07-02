"""
_common.py  --  shared helpers for the Separatrix and OW Paper figure pipeline.

Import this module FIRST in every figure script so that the matplotlib backend
is forced to Agg before any pyplot import (required for headless rendering).

Usage in a figure script:
    from _common import (
        PAPER_DIR, VFR_SRC,
        write_sidecar, compile_paper, make_parser,
    )
"""

# Force headless backend before any pyplot import.
import matplotlib
matplotlib.use("Agg")

import argparse
import hashlib
import json
import os
import platform
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

# ---------------------------------------------------------------------------
# Path constants
# ---------------------------------------------------------------------------

# This file lives at: <repo>/Paper_Writing/Separatrix and OW Paper/scripts/_common.py
_SCRIPTS_DIR = Path(__file__).resolve().parent
PAPER_DIR    = _SCRIPTS_DIR.parent
FIGURES_DIR  = PAPER_DIR / "figures"
TEX_FILE     = PAPER_DIR / "Paper_Draft_1A.tex"

# Walk up to find <repo>/trunk/Python_Simulations/Vector_Fields/VF_Robot/
_REPO_ROOT = PAPER_DIR.parent.parent
VFR_ROOT   = _REPO_ROOT / "trunk" / "Python_Simulations" / "Vector_Fields" / "VF_Robot"
VFR_SRC    = VFR_ROOT / "src"

if str(VFR_ROOT) not in sys.path:
    sys.path.insert(0, str(VFR_ROOT))


# ---------------------------------------------------------------------------
# Git helpers
# ---------------------------------------------------------------------------

def _git_commit() -> str:
    try:
        result = subprocess.run(
            ["git", "rev-parse", "--short", "HEAD"],
            capture_output=True, text=True, cwd=str(_REPO_ROOT), timeout=5,
        )
        return result.stdout.strip() if result.returncode == 0 else "unknown"
    except Exception:
        return "unknown"


def _git_dirty() -> bool:
    try:
        result = subprocess.run(
            ["git", "status", "--porcelain"],
            capture_output=True, text=True, cwd=str(_REPO_ROOT), timeout=5,
        )
        return bool(result.stdout.strip())
    except Exception:
        return False


def _file_sha1(path: Path) -> str:
    try:
        h = hashlib.sha1()
        h.update(path.read_bytes())
        return h.hexdigest()[:12]
    except Exception:
        return "unknown"


# ---------------------------------------------------------------------------
# Sidecar writer
# ---------------------------------------------------------------------------

def write_sidecar(
    png_path: Path,
    *,
    figure_name: str,
    params: dict,
    source_script: str,
    primitive_name: str = "",
    primitive_file: str = "",
    extra: dict | None = None,
) -> None:
    """
    Write a JSON provenance sidecar alongside png_path.

    Args:
        png_path:        Path to the PNG that was just written.
        figure_name:     Short name matching the figures.yaml key.
        params:          Dict of all parameters used to generate the figure.
        source_script:   Relative path of the calling script from PAPER_DIR.
        primitive_name:  Name of the control primitive used (if any).
        primitive_file:  Relative path of the primitive source file from repo root.
        extra:           Any extra scalar summaries to include (not injected into
                         the paper -- user reviews these and copies them manually).
    """
    prim_info: dict = {}
    if primitive_name:
        prim_info["primitive"] = primitive_name
    if primitive_file:
        abs_prim = _REPO_ROOT / primitive_file
        prim_info["primitive_file"] = primitive_file
        prim_info["primitive_file_sha1"] = _file_sha1(abs_prim)

    record = {
        "figure_name":    figure_name,
        "source_script":  source_script,
        "generated_utc":  datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "git_commit":     _git_commit(),
        "git_dirty":      _git_dirty(),
        "python":         platform.python_version(),
        "numpy":          np.__version__,
        "params":         params,
    }
    if prim_info:
        record["controllers"] = prim_info
    if extra:
        record["extra"] = extra

    sidecar = png_path.with_suffix(".meta.json")
    sidecar.write_text(json.dumps(record, indent=2))
    print(f"  sidecar -> {sidecar.relative_to(PAPER_DIR)}")


# ---------------------------------------------------------------------------
# LaTeX compile hook
# ---------------------------------------------------------------------------

def compile_paper(tex_path: Path = TEX_FILE) -> int:
    """
    Run two-pass pdflatex on tex_path.  Returns the pdflatex exit code of the
    second pass (0 = success).  Prints a brief status line.
    """
    cwd = tex_path.parent
    cmd = ["pdflatex", "-interaction=nonstopmode", tex_path.name]
    print(f"  compiling {tex_path.name} (pass 1) ...")
    r1 = subprocess.run(cmd, cwd=str(cwd), capture_output=True, text=True)
    print(f"  compiling {tex_path.name} (pass 2) ...")
    r2 = subprocess.run(cmd, cwd=str(cwd), capture_output=True, text=True)

    if r2.returncode != 0:
        # Print the last 20 lines of the log so the user can diagnose.
        log = (cwd / tex_path.with_suffix(".log").name)
        if log.exists():
            lines = log.read_text().splitlines()
            print("  pdflatex error -- last 20 lines of log:")
            for ln in lines[-20:]:
                print("    " + ln)
    else:
        pdf = cwd / tex_path.with_suffix(".pdf").name
        print(f"  PDF -> {pdf}")

    return r2.returncode


# ---------------------------------------------------------------------------
# Argparse template
# ---------------------------------------------------------------------------

def make_parser(figure_name: str) -> argparse.ArgumentParser:
    """
    Return an ArgumentParser pre-populated with flags common to all figure
    scripts.  The calling script may add extra arguments before calling
    parse_args().
    """
    p = argparse.ArgumentParser(
        description=f"Generate figure: {figure_name}",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument(
        "--out",
        type=Path,
        default=FIGURES_DIR / f"{figure_name}.png",
        help="Output PNG path (overwrites existing file).",
    )
    p.add_argument(
        "--no-compile",
        action="store_true",
        default=False,
        help="Skip pdflatex recompile after saving the figure.",
    )
    p.add_argument(
        "--seed",
        type=int,
        default=None,
        help="Override the random seed in PARAMS (None = use PARAMS default).",
    )
    p.add_argument(
        "--show-params",
        action="store_true",
        default=False,
        help="Print the effective PARAMS dict and exit without running.",
    )
    return p
