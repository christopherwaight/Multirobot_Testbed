#!/usr/bin/env python3
"""Render dashboard.html from status.json (run verify_revisions.py first)."""
import html
import json
from pathlib import Path

REVISION_DIR = Path(__file__).resolve().parent
STATUS_PATH = REVISION_DIR / "status.json"
OUT_PATH = REVISION_DIR / "dashboard.html"

SEV_LABEL = {"blocking": "Blocking", "should": "Should fix", "line": "Line-level"}
SEV_ORDER = ["blocking", "should", "line"]


def esc(s):
    return html.escape(str(s), quote=True)


def render_row(item):
    checked = "checked" if item["passed"] else ""
    box_cls = "done" if item["passed"] else ("waived" if item.get("waived") else "open")
    sev = item["severity"]
    sources = " ".join(f'<span class="chip src">{esc(s)}</span>' for s in item.get("sources", []))
    loc = esc(item.get("location", ""))
    note = f'<div class="note">{esc(item["note"])}</div>' if item.get("note") else ""
    evidence_cls = "ev-pass" if item["passed"] else "ev-fail"
    return f"""
    <div class="row {box_cls}" data-severity="{sev}" data-phase="{esc(item.get('phase',''))}" data-gate="{esc(item['gate_type'])}" data-group="{esc(item['group'])}">
      <div class="box"><input type="checkbox" disabled {checked}></div>
      <div class="body">
        <div class="title-line">
          <span class="idchip">{esc(item['id'])}</span>
          <span class="title">{esc(item['title'])}</span>
        </div>
        <div class="meta">
          <span class="chip gate">{esc(item['gate_type'])}</span>
          {sources}
          <span class="loc">{loc}</span>
        </div>
        {note}
        <div class="evidence {evidence_cls}">{esc(item['evidence'])}</div>
      </div>
    </div>"""


def render_group(title, items, collapsed=False):
    if not items:
        return ""
    rows = "\n".join(render_row(i) for i in items)
    open_attr = "" if not collapsed else ""
    return f"""
    <details class="group" {'' if not collapsed else 'open'}>
      <summary>{esc(title)} <span class="count">{sum(1 for i in items if i['passed'])}/{len(items)}</span></summary>
      <div class="rows">{rows}</div>
    </details>"""


def ring_svg(passed, total, color):
    pct = 0 if total == 0 else passed / total
    circumference = 2 * 3.14159265 * 40
    offset = circumference * (1 - pct)
    return f"""
    <svg viewBox="0 0 100 100" class="ring">
      <circle cx="50" cy="50" r="40" class="ring-bg"></circle>
      <circle cx="50" cy="50" r="40" class="ring-fg" style="stroke:{color};
        stroke-dasharray:{circumference:.1f};stroke-dashoffset:{offset:.1f}"></circle>
      <text x="50" y="46" class="ring-num">{passed}/{total}</text>
      <text x="50" y="62" class="ring-pct">{int(pct*100)}%</text>
    </svg>"""


def main():
    status = json.loads(STATUS_PATH.read_text())
    items = status["items"]
    generated_at = status["generated_at"]
    summary = status["summary"]

    closed = [i for i in items if i["group"] == "closed_in_6a"]
    active = [i for i in items if i["group"] != "closed_in_6a"]
    waived = [i for i in active if i.get("waived")]
    open_active = [i for i in active if not i.get("waived")]

    by_sev = {sev: [i for i in open_active if i["severity"] == sev] for sev in SEV_ORDER}

    colors = {"blocking": "#c0402a", "should": "#a67518", "line": "#3a7a5e"}
    rings = "".join(
        f'<div class="ring-wrap"><div class="ring-title">{SEV_LABEL[sev]}</div>'
        f'{ring_svg(summary[sev]["passed"], summary[sev]["total"], colors[sev])}</div>'
        for sev in SEV_ORDER
    )

    groups_html = "".join(
        render_group(SEV_LABEL[sev], by_sev[sev]) for sev in SEV_ORDER
    )
    waived_html = render_group("Waived", waived, collapsed=True) if waived else ""
    closed_html = render_group("Closed in 6a (regression guards, do not reopen)", closed, collapsed=True)

    doc = f"""<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<title>Draft 6a revision ledger</title>
<style>
:root {{
  --paper:#f1ede6; --surface:#fffdf9; --sunk:#e8e2d8;
  --ink:#1c1912; --ink2:#5a5346; --ink3:#8a8272;
  --rule:#d8d0c0; --hair:#e3ddce;
  --pass:#2f6b4f; --fail:#a8442f; --waive:#8a7530;
}}
@media (prefers-color-scheme: dark) {{
  :root {{
    --paper:#15130f; --surface:#1d1a14; --sunk:#242019;
    --ink:#ece7db; --ink2:#b8b0a0; --ink3:#847c6c;
    --rule:#3a352a; --hair:#2a261d;
    --pass:#7fd9ab; --fail:#e08a6f; --waive:#e0c070;
  }}
}}
:root[data-theme="dark"] {{
  --paper:#15130f; --surface:#1d1a14; --sunk:#242019;
  --ink:#ece7db; --ink2:#b8b0a0; --ink3:#847c6c;
  --rule:#3a352a; --hair:#2a261d;
  --pass:#7fd9ab; --fail:#e08a6f; --waive:#e0c070;
}}
:root[data-theme="light"] {{
  --paper:#f1ede6; --surface:#fffdf9; --sunk:#e8e2d8;
  --ink:#1c1912; --ink2:#5a5346; --ink3:#8a8272;
  --rule:#d8d0c0; --hair:#e3ddce;
  --pass:#2f6b4f; --fail:#a8442f; --waive:#8a7530;
}}
* {{ box-sizing: border-box; }}
body {{
  margin:0; background:var(--paper); color:var(--ink);
  font-family:-apple-system,BlinkMacSystemFont,"Segoe UI",sans-serif;
  font-size:15px; line-height:1.5;
}}
.wrap {{ max-width:980px; margin:0 auto; padding:28px 20px 80px; }}
h1 {{ font-size:22px; margin:0 0 4px; letter-spacing:-.01em; }}
.sub {{ color:var(--ink3); font-size:13px; margin:0 0 24px; font-family:ui-monospace,monospace; }}
.rings {{ display:flex; gap:24px; margin-bottom:28px; flex-wrap:wrap; }}
.ring-wrap {{ text-align:center; }}
.ring-title {{ font-size:12px; color:var(--ink2); margin-bottom:4px; text-transform:uppercase; letter-spacing:.06em; }}
.ring {{ width:96px; height:96px; }}
.ring-bg {{ fill:none; stroke:var(--hair); stroke-width:8; }}
.ring-fg {{ fill:none; stroke-width:8; stroke-linecap:round; transform:rotate(-90deg); transform-origin:50% 50%; transition:stroke-dashoffset .3s; }}
.ring-num {{ font-size:20px; text-anchor:middle; fill:var(--ink); font-weight:600; }}
.ring-pct {{ font-size:10px; text-anchor:middle; fill:var(--ink3); }}
details.group {{ background:var(--surface); border:1px solid var(--rule); border-radius:10px; margin-bottom:14px; overflow:hidden; }}
summary {{ padding:14px 18px; cursor:pointer; font-weight:600; list-style:none; display:flex; justify-content:space-between; }}
summary::-webkit-details-marker {{ display:none; }}
summary::before {{ content:"▸ "; color:var(--ink3); }}
details[open] summary::before {{ content:"▾ "; }}
.count {{ color:var(--ink3); font-weight:400; font-family:ui-monospace,monospace; }}
.rows {{ border-top:1px solid var(--hair); }}
.row {{ display:flex; gap:12px; padding:12px 18px; border-bottom:1px solid var(--hair); }}
.row:last-child {{ border-bottom:none; }}
.row.done {{ opacity:.6; }}
.row.waived {{ opacity:.5; }}
.row.waived .title {{ text-decoration:line-through; }}
.box input {{ width:16px; height:16px; margin-top:2px; }}
.body {{ flex:1; min-width:0; }}
.title-line {{ display:flex; gap:8px; align-items:baseline; flex-wrap:wrap; }}
.idchip {{ font-family:ui-monospace,monospace; font-size:11px; background:var(--sunk); color:var(--ink2); padding:1px 6px; border-radius:4px; }}
.title {{ font-weight:500; }}
.meta {{ display:flex; gap:6px; flex-wrap:wrap; margin-top:6px; align-items:center; }}
.chip {{ font-size:10.5px; padding:1px 7px; border-radius:9px; font-family:ui-monospace,monospace; }}
.chip.gate {{ background:var(--sunk); color:var(--ink2); }}
.chip.src {{ background:transparent; border:1px solid var(--rule); color:var(--ink3); }}
.loc {{ font-size:11.5px; color:var(--ink3); font-family:ui-monospace,monospace; }}
.note {{ font-size:12.5px; color:var(--ink2); margin-top:6px; font-style:italic; }}
.evidence {{ font-size:12px; font-family:ui-monospace,monospace; margin-top:6px; padding:4px 8px; border-radius:5px; }}
.ev-pass {{ background:color-mix(in srgb, var(--pass) 12%, transparent); color:var(--pass); }}
.ev-fail {{ background:color-mix(in srgb, var(--fail) 10%, transparent); color:var(--fail); }}
.filters {{ display:flex; gap:8px; margin-bottom:20px; flex-wrap:wrap; }}
.filters button {{
  font-family:inherit; font-size:12.5px; padding:5px 12px; border-radius:14px;
  border:1px solid var(--rule); background:var(--surface); color:var(--ink2); cursor:pointer;
}}
.filters button.active {{ background:var(--ink); color:var(--paper); border-color:var(--ink); }}
</style>
</head>
<body>
<div class="wrap">
  <h1>Draft 6a revision ledger</h1>
  <p class="sub">generated {esc(generated_at)} &middot; gates: tex / number / script / manual, per revision/items.yaml</p>
  <div class="rings">{rings}</div>
  <div class="filters" id="filters">
    <button data-f="all" class="active">All</button>
    <button data-f="blocking">Blocking</button>
    <button data-f="should">Should fix</button>
    <button data-f="line">Line-level</button>
    <button data-f="fail">Open only</button>
    <button data-f="number">Number gates</button>
  </div>
  {groups_html}
  {waived_html}
  {closed_html}
</div>
<script>
document.getElementById('filters').addEventListener('click', (e) => {{
  const btn = e.target.closest('button');
  if (!btn) return;
  document.querySelectorAll('#filters button').forEach(b => b.classList.remove('active'));
  btn.classList.add('active');
  const f = btn.dataset.f;
  document.querySelectorAll('.row').forEach(row => {{
    let show = true;
    if (f === 'blocking' || f === 'should' || f === 'line') show = row.dataset.severity === f;
    else if (f === 'fail') show = !row.classList.contains('done');
    else if (f === 'number') show = row.dataset.gate === 'number';
    row.style.display = show ? '' : 'none';
  }});
}});
</script>
</body>
</html>"""

    OUT_PATH.write_text(doc)
    print(f"Wrote {OUT_PATH}")


if __name__ == "__main__":
    main()
