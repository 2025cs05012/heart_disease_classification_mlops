#!/usr/bin/env bash
# Renders reports/architecture.mmd to screenshots/architecture.png via Mermaid CLI.
set -e
cd "$(dirname "$0")/.."
export PUPPETEER_SKIP_DOWNLOAD=true
export PUPPETEER_SKIP_CHROMIUM_DOWNLOAD=true
exec npx --yes -p @mermaid-js/mermaid-cli mmdc \
    -i reports/architecture.mmd \
    -o screenshots/architecture.png \
    -p reports/.puppeteer.json \
    -w 2400 -H 1500 -s 2 -b white
