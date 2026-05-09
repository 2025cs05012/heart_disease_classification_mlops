#!/usr/bin/env bash
# Rebuilds the architecture PNGs + REPORT.html / REPORT.pdf / REPORT.docx
# from the canonical Markdown sources. Run after editing
# reports/REPORT.md or reports/architecture.md.
#
# Requirements (already on dev machine):
#   - pandoc                 (brew install pandoc)
#   - npx + node             (any recent node)
#   - Google Chrome          (used as headless PDF engine)
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
REPORTS="$REPO_ROOT/Assignment/reports"
FIGURES="$REPORTS/figures"
CHROME="/Applications/Google Chrome.app/Contents/MacOS/Google Chrome"

cd "$REPO_ROOT"

echo "==> 1. Re-render Mermaid diagrams"
python3 "$REPORTS/_extract_mmd.py"
npx -y -p @mermaid-js/mermaid-cli mmdc \
    -i "$REPORTS/_arch_flow.mmd" \
    -o "$FIGURES/architecture.png" \
    -t neutral -b white -w 1600 -H 1000
npx -y -p @mermaid-js/mermaid-cli mmdc \
    -i "$REPORTS/_arch_seq.mmd" \
    -o "$FIGURES/architecture_sequence.png" \
    -t neutral -b white -w 1400 -H 900
rm -f "$REPORTS/_arch_flow.mmd" "$REPORTS/_arch_seq.mmd"

echo "==> 2. Markdown -> standalone HTML (CSS + images embedded)"
pandoc "$REPORTS/REPORT.md" -s \
    --metadata title="Heart Disease MLOps — Final Report" \
    --resource-path="$REPORTS:$FIGURES" \
    --embed-resources --standalone \
    -c "$REPORTS/_report.css" \
    -o "$REPORTS/REPORT.html"

echo "==> 3. Markdown -> DOCX"
pandoc "$REPORTS/REPORT.md" \
    --resource-path="$REPORTS:$FIGURES" \
    -o "$REPORTS/REPORT.docx"

echo "==> 4. HTML -> PDF (headless Chrome)"
"$CHROME" \
    --headless=new --disable-gpu --no-sandbox \
    --print-to-pdf="$REPORTS/REPORT.pdf" \
    --print-to-pdf-no-header \
    --no-pdf-header-footer \
    --virtual-time-budget=10000 \
    "file://$REPORTS/REPORT.html" 2>&1 | tail -1

echo "==> Done."
ls -la "$REPORTS/REPORT.html" "$REPORTS/REPORT.pdf" "$REPORTS/REPORT.docx" \
       "$FIGURES/architecture.png" "$FIGURES/architecture_sequence.png"
