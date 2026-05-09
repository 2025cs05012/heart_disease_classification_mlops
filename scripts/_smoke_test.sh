#!/bin/bash
set -e
DOCKER=/usr/local/bin/docker
PORT=8088
NAME=heartdemo

# clean up any previous run
$DOCKER rm -f $NAME 2>/dev/null || true

echo "==> starting container heart-api:latest on host port $PORT"
$DOCKER run --rm -d -p $PORT:5000 --name $NAME heart-api:latest >/dev/null
sleep 4

echo
echo "===== /health ====="
curl -s -w "  HTTP %{http_code}\n" http://localhost:$PORT/health

echo
echo "===== / (first 8 lines + form fields) ====="
curl -s http://localhost:$PORT/ | head -8
echo "..."
curl -s http://localhost:$PORT/ | grep -oE 'name="[a-z_]+"' | sort -u | head -20

echo
echo "===== /predict (disease-risk sample) ====="
curl -s -X POST http://localhost:$PORT/predict \
  -H 'Content-Type: application/json' \
  -d '{"age":63,"sex":1,"cp":3,"trestbps":145,"chol":233,"fbs":1,"restecg":0,"thalach":150,"exang":0,"oldpeak":2.3,"slope":0,"ca":0,"thal":1,"ca_missing":0}'
echo

echo
echo "===== /predict (low-risk sample) ====="
curl -s -X POST http://localhost:$PORT/predict \
  -H 'Content-Type: application/json' \
  -d '{"age":29,"sex":0,"cp":1,"trestbps":120,"chol":180,"fbs":0,"restecg":0,"thalach":190,"exang":0,"oldpeak":0.0,"slope":2,"ca":0,"thal":2,"ca_missing":0}'
echo

echo
echo "===== screenshot via headless Chrome ====="
SHOT=/Users/nekka/Documents/nekka/Project/SmfCodes/Branch_Check/2026_4/smf-cd-pipeline/Assignment/screenshots/task6_form.png
mkdir -p "$(dirname "$SHOT")"
"/Applications/Google Chrome.app/Contents/MacOS/Google Chrome" \
  --headless=new --disable-gpu --hide-scrollbars \
  --window-size=1280,1100 \
  --screenshot="$SHOT" \
  "http://localhost:$PORT/" 2>/dev/null
ls -lh "$SHOT"

echo
echo "===== container logs (last 12 lines) ====="
$DOCKER logs --tail 12 $NAME 2>&1

echo
echo "==> cleaning up"
$DOCKER rm -f $NAME >/dev/null
echo "==> SMOKE TEST PASSED"
