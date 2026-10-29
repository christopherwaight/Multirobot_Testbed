"""
Build a self-contained HTML version of hyperparameter_study.md, with every
figure embedded as a data URI (one file, nothing external but Google Fonts).

    venv/bin/python3 experiments/hyperparameter_study/build_report.py
Writes hyperparameter_study.html next to the markdown.

Handles the markdown subset the report uses: #/##/### headings, paragraphs,
**bold**, `code`, [links](url), ![figures](path), - lists, ``` blocks, tables.
"""
import base64
import html
import os
import re

HERE = os.path.dirname(os.path.abspath(__file__))
SRC = os.path.join(HERE, "hyperparameter_study.md")
OUT = os.path.join(HERE, "hyperparameter_study.html")

CSS = """
<title>Section IV Hyperparameter Study</title>
<meta name="claude:viewport-fit-sides" content="cover">
<link rel="preconnect" href="https://fonts.googleapis.com">
<link rel="stylesheet" href="https://fonts.googleapis.com/css2?family=IBM+Plex+Sans:ital,wght@0,400;0,500;0,600;1,400&family=IBM+Plex+Sans+Condensed:wght@500;600&family=IBM+Plex+Mono:wght@400;500&display=swap">
<style>
/* Layout: one reading column for prose, wider plates for figures; formulas as instrument panels. */
:root {
  --bg: #f7f8fa; --fg: #15181c; --muted: #555c66; --rule: #d9dde3;
  --accent: #1c5cab; --panel: #eef1f5; --plate: #ffffff; --plate-edge: #d4d9e0;
  --display: "IBM Plex Sans Condensed", "Arial Narrow", system-ui, sans-serif;
  --body: "IBM Plex Sans", system-ui, -apple-system, "Segoe UI", sans-serif;
  --mono: "IBM Plex Mono", ui-monospace, "SF Mono", Menlo, monospace;
}
@media (prefers-color-scheme: dark) { :root:not([data-theme="light"]) {
  --bg: #111418; --fg: #e4e7ec; --muted: #a2a9b4; --rule: #2c323a;
  --accent: #7fb0f0; --panel: #1a1f26; --plate: #ffffff; --plate-edge: #3a414b; color-scheme: dark } }
:root[data-theme="dark"] {
  --bg: #111418; --fg: #e4e7ec; --muted: #a2a9b4; --rule: #2c323a;
  --accent: #7fb0f0; --panel: #1a1f26; --plate: #ffffff; --plate-edge: #3a414b; color-scheme: dark }
body { background: var(--bg); color: var(--fg); font: 15px/1.6 var(--body); }
main { max-width: 1120px; margin: 0 auto; padding-block: 40px 72px;
  padding-left: max(16px, env(safe-area-inset-left, 0px)); padding-right: max(16px, env(safe-area-inset-right, 0px)); }
.prose { max-width: 70ch; }
main > p, main > ul, main > h2, main > h3 { max-width: 78ch; }
main > pre { max-width: min(110ch, 100%); box-sizing: border-box; }
h1 { font: 600 2.1rem/1.15 var(--display); letter-spacing: -0.01em; margin: 0 0 .4rem; text-wrap: balance; }
h2 { font: 600 1.35rem/1.25 var(--display); margin: 2.6rem 0 .6rem; padding-top: 1.1rem; border-top: 1px solid var(--rule); text-wrap: balance; }
h3 { font: 600 1.05rem/1.3 var(--display); margin: 1.6rem 0 .4rem; }
p { margin: 0 0 .9rem; }
.lede { color: var(--muted); font-size: .95rem; }
strong { font-weight: 600; }
a { color: var(--accent); }
code { font: .86em var(--mono); background: var(--panel); padding: .08em .3em; border-radius: 3px; }
pre { font: .84rem/1.55 var(--mono); background: var(--panel); border-left: 3px solid var(--accent);
  padding: .8rem 1rem; margin: 0 0 1.1rem; overflow-x: auto; max-width: 100%; }
pre code { background: none; padding: 0; }
ul { margin: 0 0 1rem; padding-left: 1.2rem; display: grid; gap: .35rem; }
figure { margin: 1.2rem 0 1.8rem; }
figure img { display: block; width: 100%; height: auto; background: var(--plate);
  border: 1px solid var(--plate-edge); border-radius: 4px; padding: 6px; box-sizing: border-box; }
figcaption { font-size: .85rem; color: var(--muted); margin-top: .45rem; max-width: 90ch; }
.table-wrap { overflow-x: auto; margin: 0 0 1.4rem; max-width: 100%; }
table { border-collapse: collapse; font-size: .84rem; font-variant-numeric: tabular-nums; min-width: 100%; }
th, td { text-align: left; padding: .4rem .6rem; border-bottom: 1px solid var(--rule); vertical-align: top; }
th { font: 600 .74rem/1.3 var(--body); text-transform: uppercase; letter-spacing: .04em; color: var(--muted);
  border-bottom: 1.5px solid var(--fg); }
td:first-child { font-weight: 500; }
.tablecap { font-size: .9rem; margin: 1.2rem 0 .4rem; }
a:focus-visible { outline: 2px solid var(--accent); outline-offset: 2px; }
@media (max-width: 600px) { h1 { font-size: 1.6rem; } body { font-size: 14px; } }
</style>
"""


def inline(t):
    t = html.escape(t, quote=False)
    t = re.sub(r"`([^`]+)`", r"<code>\1</code>", t)
    t = re.sub(r"\*\*([^*]+)\*\*", r"<strong>\1</strong>", t)
    t = re.sub(r"\[([^\]]+)\]\(([^)]+)\)", r'<a href="\2">\1</a>', t)
    return t


def figure(alt, path):
    full = os.path.join(HERE, path)
    data = base64.b64encode(open(full, "rb").read()).decode()
    return (f'<figure><img alt="{html.escape(alt)}" src="data:image/png;base64,{data}">'
            f'<figcaption>{inline(alt)}</figcaption></figure>')


def convert(md):
    lines = md.splitlines()
    out, i, para = [], 0, []

    def flush():
        if para:
            out.append(f'<p>{inline(" ".join(para))}</p>')
            para.clear()
    while i < len(lines):
        ln = lines[i]
        s = ln.strip()
        if s.startswith("```"):
            flush()
            j = i + 1
            block = []
            while j < len(lines) and not lines[j].strip().startswith("```"):
                block.append(lines[j])
                j += 1
            out.append("<pre><code>" + html.escape("\n".join(block)) + "</code></pre>")
            i = j + 1
            continue
        m = re.match(r"^(#{1,3}) (.*)$", s)
        if m:
            flush()
            n = len(m.group(1))
            out.append(f"<h{n}>{inline(m.group(2))}</h{n}>")
            i += 1
            continue
        m = re.match(r"^!\[(.*)\]\((.*)\)$", s)
        if m:
            flush()
            out.append(figure(m.group(1), m.group(2)))
            i += 1
            continue
        if s.startswith("|"):
            flush()
            rows = []
            while i < len(lines) and lines[i].strip().startswith("|"):
                rows.append([c.strip() for c in lines[i].strip().strip("|").split("|")])
                i += 1
            head, body = rows[0], [r for r in rows[1:] if not all(set(c) <= set("-: ") for c in r)]
            t = ["<div class=\"table-wrap\"><table><thead><tr>"] + [f"<th>{inline(c)}</th>" for c in head] + \
                ["</tr></thead><tbody>"]
            for r in body:
                t.append("<tr>" + "".join(f"<td>{inline(c)}</td>" for c in r) + "</tr>")
            t.append("</tbody></table></div>")
            out.append("".join(t))
            continue
        if s.startswith("- "):
            flush()
            items = []
            while i < len(lines) and lines[i].strip().startswith("- "):
                item = [lines[i].strip()[2:]]
                i += 1
                while i < len(lines) and lines[i].startswith("  ") and lines[i].strip() and \
                        not lines[i].strip().startswith("- "):
                    item.append(lines[i].strip())
                    i += 1
                items.append(" ".join(item))
            out.append("<ul>" + "".join(f"<li>{inline(x)}</li>" for x in items) + "</ul>")
            continue
        if not s:
            flush()
        else:
            para.append(s)
        i += 1
    flush()
    return "\n".join(out)


def main():
    body = convert(open(SRC).read())
    body = body.replace("<h1>", "<header class=\"prose\"><h1>", 1)
    first_p = body.find("<p>", body.find("</h1>"))
    end_p = body.find("</p>", first_p) + 4
    body = body[:first_p] + body[first_p:end_p].replace("<p>", '<p class="lede">', 1) + "</header>" + body[end_p:]
    page = CSS + "\n<main>\n" + body + "\n</main>\n"
    open(OUT, "w").write(page)
    print(f"wrote {OUT} ({len(page) / 1e6:.1f} MB)")


if __name__ == "__main__":
    main()
