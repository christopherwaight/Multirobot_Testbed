"""Verbatim porting helper for the thesis chapters.

Each chapter script builds a list of parts. P() copies a passage from a
source .tex, located by the text of its first and last lines, and applies
only the listed exact substitutions. Any anchor or substitution that does
not match raises, so a drifted source aborts the build instead of
silently porting the wrong text. T() is literal text (new prose, figure
environments, or IDETC text transcribed from the published PDF).
"""
import os, re, sys

ROOT = '/Users/christopherwaight/Desktop/Multirobot_Testbed/Paper_Writing/'
SRC = {
    'sys': ROOT + 'Systems Paper/cwaight_systems_paper/cwaight_systems_paper.tex',
    'sep': ROOT + 'Separatrix_and_OW_Paper/Draft_11.tex',
}
CH = ROOT + 'PhD Thesis/chapters/'
_cache = {}

def _lines(src):
    if src not in _cache:
        _cache[src] = open(SRC[src]).read().split('\n')
    return _cache[src]

class PortError(Exception):
    pass

PRE = None  # chapter label prefix, set by each chapter script

def _prefix(text, pre):
    # eq:foo -> eq:ch6:foo for single-colon labels only; labels that
    # already name a chapter (eq:ch3:foo) or ch:/app: labels are left alone.
    def fix(m):
        cmd, kind, name = m.group(1), m.group(2), m.group(3)
        return f'\\{cmd}{{{kind}:{pre}:{name}}}'
    return re.sub(r'\\(label|ref|eqref)\{(eq|fig|tab|sec):([^:}]+)\}', fix, text)

def P(src, start, end=None, subs=(), cites=True, tag=None, back=0, extra=0):
    """Copy lines from the one containing `start` through the first line
    at or after it containing `end` (default: the start line only).
    `back` and `extra` widen the passage by whole lines."""
    L = _lines(src)
    hits = [i for i, l in enumerate(L) if start in l]
    if len(hits) != 1:
        raise PortError(f'{src}: start anchor matched {len(hits)} lines: {start!r}')
    i = hits[0]
    j = i
    if end is not None:
        js = [k for k in range(i, len(L)) if end in L[k]]
        if not js:
            raise PortError(f'{src}: end anchor not found after start: {end!r}')
        j = js[0]
    text = '\n'.join(L[i - back:j + 1 + extra])
    text = text.replace('[!t]', '[htbp]').replace('[!tb]', '[htbp]')
    for old, new in subs:
        n = text.count(old)
        if n == 0:
            raise PortError(f'{src}: substitution not found: {old!r}')
        text = text.replace(old, new)
    if cites:
        text = re.sub(r'\\cite\{([^}]*)\}',
                      lambda m: '\\cite{' + ','.join(
                          k if ':' in k else f'{src}:{k.strip()}'
                          for k in m.group(1).split(',')) + '}', text)
    if PRE:
        text = _prefix(text, PRE)
    head = f'% Ported from: {os.path.basename(SRC[src])}' + (f', {tag}' if tag else '')
    return head + '\n' + text.strip('\n') + '\n'

def T(text):
    return text.strip('\n') + '\n'

def write(name, parts):
    out = '\n'.join(p for p in parts)
    path = CH + name
    open(path, 'w').write(out)
    print(f'wrote {name}: {len(out.split())} tokens')
