#!/bin/bash
# Capture the form page via headless Chrome with all the working flags
# (foreground, with explicit timeout, plus --virtual-time-budget for predictable rendering).
SHOT=/Users/nekka/Documents/nekka/Project/SmfCodes/Branch_Check/2026_4/smf-cd-pipeline/Assignment/screenshots/task6_form.png
TMPDIR=$(mktemp -d)
mkdir -p "$(dirname "$SHOT")"

CHROME_BIN="/Applications/Google Chrome.app/Contents/MacOS/Google Chrome"

# Run chrome in a clean subshell with stdin closed; bound by /usr/bin/timeout-like behavior using `&` + `wait` + sleep-based timeout.
(
  "$CHROME_BIN" \
    --headless=new \
    --disable-gpu \
    --hide-scrollbars \
    --no-sandbox \
    --no-first-run \
    --no-default-browser-check \
    --disable-extensions \
    --disable-component-update \
    --disable-background-networking \
    --user-data-dir="$TMPDIR" \
    --window-size=1280,1200 \
    --virtual-time-budget=4000 \
    --screenshot="$SHOT" \
    "http://localhost:8088/" </dev/null >/tmp/chrome.log 2>&1
) &
CPID=$!

# wait up to 30s polling for the screenshot
for i in $(seq 1 30); do
  if [ -s "$SHOT" ]; then break; fi
  sleep 1
done

# kill chrome if still alive
kill $CPID 2>/dev/null
wait $CPID 2>/dev/null

rm -rf "$TMPDIR"
ls -lh "$SHOT" 2>&1
echo "===chrome.log tail==="
tail -10 /tmp/chrome.log 2>&1
