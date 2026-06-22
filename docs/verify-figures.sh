#!/usr/bin/env bash
# verify-figures.sh -- report missing \includegraphics files in a .tex file
# Usage: bash docs/verify-figures.sh "path/to/paper.tex"
# Read-only. No changes are made.

set -euo pipefail

TEX_FILE="${1:?Usage: verify-figures.sh path/to/paper.tex}"

if [ ! -f "$TEX_FILE" ]; then
  echo "Error: file not found: $TEX_FILE"
  exit 1
fi

TEX_DIR="$(dirname "$TEX_FILE")"

echo "Checking figures in: $TEX_FILE"
echo "Resolved relative to: $TEX_DIR"
echo "---"

MISSING=0
FOUND=0

while IFS= read -r line; do
  # Strip everything before \includegraphics, then extract the first {filename}
  # This handles \includegraphics{f}, \includegraphics[opts]{f}, and
  # nested forms like \IEEEbiography[{\includegraphics[opts]{f}}]{Author Name}
  after=$(echo "$line" | sed 's/.*\\includegraphics//')
  # Skip optional [...] block if present
  after=$(echo "$after" | sed 's/^\[[^]]*\]//')
  # Now the first {filename} is what remains
  fname=$(echo "$after" | grep -oE '^\{[^}]+\}' | tr -d '{}')
  [ -z "$fname" ] && continue

  resolved=""
  if [ -f "$TEX_DIR/$fname" ]; then
    resolved="$TEX_DIR/$fname"
  else
    for ext in .png .jpg .jpeg .pdf .eps; do
      if [ -f "$TEX_DIR/$fname$ext" ]; then
        resolved="$TEX_DIR/$fname$ext"
        break
      fi
    done
  fi

  if [ -n "$resolved" ]; then
    echo "OK     $fname"
    FOUND=$((FOUND + 1))
  else
    echo "MISS   $fname"
    MISSING=$((MISSING + 1))
  fi
done < <(grep -n '\\includegraphics' "$TEX_FILE")

echo "---"
echo "Found: $FOUND  Missing: $MISSING"
[ "$MISSING" -gt 0 ] && exit 1 || exit 0
