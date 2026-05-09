#!/usr/bin/env bash
# Renders reports/architecture.mmd to reports/architecture.pdf via Mermaid CLI,
# using the system Chrome instead of downloading a fresh chromium.
set -e
cd "$(dirname "$0")/.."
export PUPPETEER_SKIP_DOWNLOAD=true
export PUPPETEER_SKIP_CHROMIUM_DOWNLOAD=true
exec npx --yes -p @mermaid-js/mermaid-cli mmdc \
    -i reports/architecture.mmd \
    -o reports/architecture.pdf \
    -p reports/.puppeteer.json \
    -w 2400 -H 1500 --pdfFit -b white
