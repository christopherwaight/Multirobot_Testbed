#!/usr/bin/env python3
"""Verify the Draft_6a revision ledger (items.yaml) against the manuscript and
repo artifacts, and write status.json.

Read-only against Draft_6a.tex and every simulation artifact it references.
Never writes into the manuscript.

Usage:
    python3 verify_revisions.py [--json-only] [--filter ID,ID,...] [--strict]

Gate types:
    tex     - regex assertions against Draft_6a.tex
    number  - generator script exists (git-tracked) and is older than its
              artifact, artifact resolves `key`, and the value the tex_pattern
              captures from Draft_6a.tex matches within rtol
    script  - generator + artifact exist, artifact newer than generator and
              (if given) newer than a watched source file
    manual  - not machine-checkable; status comes only from waivers.yaml
"""
import argparse
import json
import re
import subprocess
import sys
from pathlib import Path

REVISION_DIR = Path(__file__).resolve().parent
PAPER_DIR = REVISION_DIR.parent
REPO_ROOT = PAPER_DIR.parent.parent  # Paper_Writing/Separatrix_and_OW_Paper -> repo root
TEX_PATH = PAPER_DIR / "Draft_6a.tex"
ITEMS_PATH = REVISION_DIR / "items.yaml"
WAIVERS_PATH = REVISION_DIR / "waivers.yaml"
STATUS_PATH = REVISION_DIR / "status.json"

try:
    import yaml
except ImportError:
    print("PyYAML is required. Install with: pip install pyyaml", file=sys.stderr)
    sys.exit(2)


def load_yaml(path):
    if not path.exists():
        return {}
    with open(path) as f:
        return yaml.safe_load(f) or {}


def load_tex():
    return TEX_PATH.read_text(encoding="utf-8")


def is_git_tracked(path: Path) -> bool:
    try:
        rel = path.resolve().relative_to(REPO_ROOT.resolve())
    except ValueError:
        return False
    result = subprocess.run(
        ["git", "ls-files", "--error-unmatch", str(rel)],
        cwd=REPO_ROOT, capture_output=True, text=True,
    )
    return result.returncode == 0


def resolve_key(obj, dotted_key):
    cur = obj
    for part in dotted_key.split("."):
        if isinstance(cur, dict) and part in cur:
            cur = cur[part]
        else:
            return None
    return cur


def check_tex_gate(gate, tex):
    flags = re.MULTILINE | re.DOTALL
    checks = []

    def rx(pattern):
        return re.search(pattern, tex, flags)

    if "present" in gate:
        ok = rx(gate["present"]) is not None
        checks.append((f"present:{gate['present'][:60]}", ok))
    if "absent" in gate:
        ok = rx(gate["absent"]) is None
        checks.append((f"absent:{gate['absent'][:60]}", ok))
    if "present_all" in gate:
        for pat in gate["present_all"]:
            ok = rx(pat) is not None
            checks.append((f"present:{pat[:60]}", ok))
    if "absent_all" in gate:
        for pat in gate["absent_all"]:
            ok = rx(pat) is None
            checks.append((f"absent:{pat[:60]}", ok))

    passed = all(ok for _, ok in checks) if checks else False
    evidence = "; ".join(f"{'OK' if ok else 'FAIL'} {label}" for label, ok in checks)
    return passed, evidence or "no assertions defined"


def resolve_csv_value(artifact_path, row_match, column):
    import csv
    with open(artifact_path, newline="") as f:
        for row in csv.DictReader(f):
            if all(row.get(k) == v for k, v in row_match.items()):
                return row.get(column)
    return None


def check_number_gate(gate):
    generator = REPO_ROOT / gate["generator"]
    artifact = REPO_ROOT / gate["artifact"]

    if not generator.exists():
        return False, f"NO_GENERATOR ({gate['generator']} does not exist)"
    if not is_git_tracked(generator):
        return False, f"NO_GENERATOR ({gate['generator']} exists but is not git-tracked)"
    if not artifact.exists():
        return False, f"NO_GENERATOR (artifact {gate['artifact']} does not exist yet)"
    if artifact.stat().st_mtime < generator.stat().st_mtime:
        return False, "STALE (artifact older than generator; re-run needed)"

    if artifact.suffix.lower() == ".csv":
        row_match = gate.get("row_match", {})
        column = gate.get("column")
        if not column:
            return False, "KEY_MISSING (csv artifact needs 'column' in gate)"
        value = resolve_csv_value(artifact, row_match, column)
        key_desc = f"{row_match}.{column}"
    else:
        try:
            data = json.loads(artifact.read_text())
        except (json.JSONDecodeError, UnicodeDecodeError) as e:
            return False, f"KEY_MISSING (artifact not valid JSON: {e})"
        value = resolve_key(data, gate["key"])
        key_desc = gate["key"]

    if value is None:
        return False, f"KEY_MISSING (key '{key_desc}' not found in artifact)"

    tex = load_tex()
    m = re.search(gate["tex_pattern"], tex, re.MULTILINE | re.DOTALL)
    if not m:
        return False, f"TEX_MISSING (pattern not found in Draft_6a.tex: {gate['tex_pattern'][:60]})"

    try:
        tex_value = float(m.group(1))
    except (ValueError, IndexError):
        return False, f"TEX_MISSING (pattern matched but no numeric capture group)"

    try:
        artifact_value = float(value) * gate.get("scale", 1.0)
    except (TypeError, ValueError):
        return False, f"KEY_MISSING (key '{key_desc}' resolved to non-numeric: {value!r})"

    rtol = gate.get("rtol", 0.02)
    if artifact_value == 0:
        ok = abs(tex_value) < 1e-9
    else:
        ok = abs(tex_value - artifact_value) / abs(artifact_value) <= rtol
    if ok:
        return True, f"MATCH (tex {tex_value} ~= artifact {artifact_value}, rtol {rtol})"
    return False, f"MISMATCH (tex {tex_value} vs artifact {artifact_value}, rtol {rtol})"


def check_script_gate(gate):
    generator = REPO_ROOT / gate["generator"]
    artifact = REPO_ROOT / gate["artifact"]

    if not generator.exists():
        return False, f"NO_GENERATOR ({gate['generator']} does not exist)"
    if not is_git_tracked(generator):
        return False, f"NO_GENERATOR ({gate['generator']} exists but is not git-tracked)"
    if not artifact.exists():
        return False, f"NO_GENERATOR (artifact {gate['artifact']} does not exist yet)"
    if artifact.stat().st_size == 0:
        return False, "STALE (artifact is empty)"
    if artifact.stat().st_mtime < generator.stat().st_mtime:
        return False, "STALE (artifact older than generator; re-run needed)"

    watch = gate.get("watch")
    if watch:
        watch_path = REPO_ROOT / watch
        if watch_path.exists() and artifact.stat().st_mtime < watch_path.stat().st_mtime:
            return False, f"STALE (artifact older than watched file {watch}; re-run needed)"

    return True, f"PRESENT ({gate['artifact']}, newer than generator)"


def check_item(item, tex):
    gate = item.get("gate", {})
    gtype = gate.get("type", "manual")

    if gtype == "tex":
        return check_tex_gate(gate, tex)
    if gtype == "number":
        return check_number_gate(gate)
    if gtype == "script":
        return check_script_gate(gate)
    if gtype == "manual":
        return False, "manual (no machine gate; see waivers.yaml)"
    return False, f"unknown gate type: {gtype}"


def iter_all_items(items_doc):
    for group_name, group in items_doc.items():
        if not isinstance(group, list):
            continue
        for item in group:
            item = dict(item)
            item["_group"] = group_name
            yield item


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--json-only", action="store_true", help="write status.json, skip the dashboard")
    parser.add_argument("--filter", help="comma-separated item IDs to verify")
    parser.add_argument("--strict", action="store_true", help="exit 1 if any blocking item fails")
    args = parser.parse_args()

    items_doc = load_yaml(ITEMS_PATH)
    waivers_doc = load_yaml(WAIVERS_PATH)
    waivers = {w["id"]: w for w in waivers_doc.get("waivers", [])} if waivers_doc else {}

    tex = load_tex()
    filter_ids = set(args.filter.split(",")) if args.filter else None

    results = []
    for item in iter_all_items(items_doc):
        item_id = item["id"]
        if filter_ids and item_id not in filter_ids:
            continue

        waiver = waivers.get(item_id)
        if waiver:
            passed = waiver.get("status") == "done"
            evidence = f"WAIVED: {waiver.get('reason', 'no reason given')}"
        else:
            passed, evidence = check_item(item, tex)

        results.append({
            "id": item_id,
            "title": item.get("title", ""),
            "sources": item.get("sources", []),
            "severity": item.get("severity", "line"),
            "phase": item.get("phase"),
            "location": item.get("location", ""),
            "note": item.get("note", ""),
            "group": item["_group"],
            "gate_type": item.get("gate", {}).get("type", "manual"),
            "passed": passed,
            "evidence": evidence,
            "waived": waiver is not None,
        })

    status = {
        "generated_at": __import__("datetime").datetime.now().isoformat(timespec="seconds"),
        "items": results,
        "summary": {},
    }
    for sev in ("blocking", "should", "line"):
        sev_items = [r for r in results if r["severity"] == sev]
        status["summary"][sev] = {
            "total": len(sev_items),
            "passed": sum(1 for r in sev_items if r["passed"]),
        }

    STATUS_PATH.write_text(json.dumps(status, indent=2))

    total = len(results)
    passed = sum(1 for r in results if r["passed"])
    print(f"Verified {total} items: {passed} passed, {total - passed} open.")
    for sev in ("blocking", "should", "line"):
        s = status["summary"][sev]
        print(f"  {sev:9s}: {s['passed']}/{s['total']}")

    failing_details = [r for r in results if not r["passed"]]
    if failing_details:
        print("\nOpen items:")
        for r in failing_details:
            print(f"  [{r['severity']:8s}] {r['id']:28s} {r['evidence']}")

    if not args.json_only:
        if filter_ids:
            print("\n(--filter was used: status.json/dashboard now reflect only the "
                  "filtered subset. Re-run without --filter before publishing.)")
        build_dashboard = REVISION_DIR / "build_dashboard.py"
        if build_dashboard.exists():
            subprocess.run([sys.executable, str(build_dashboard)], check=True)

    if args.strict:
        blocking_fail = any(r["severity"] == "blocking" and not r["passed"] for r in results)
        if blocking_fail:
            print("\nSTRICT MODE: blocking items still open.", file=sys.stderr)
            sys.exit(1)


if __name__ == "__main__":
    main()
