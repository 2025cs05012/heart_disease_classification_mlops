#!/usr/bin/env bash
# Generate ~30s of mixed traffic against the heart-api so every panel on
# the Grafana "Heart API" dashboard shows a visible spike. Useful for
# screen-recording a demo or sanity-checking the full
# Prometheus -> Grafana -> heart-api scrape path end-to-end.
#
# Targets http://localhost/ (the Ingress URL exposed by demo_up.sh).
# Override with API_URL=... if you front the API differently.
set -euo pipefail

API_URL="${API_URL:-http://localhost}"

DISEASE='{"age":63,"sex":1,"cp":3,"trestbps":145,"chol":233,"fbs":1,"restecg":0,"thalach":150,"exang":0,"oldpeak":2.3,"slope":0,"ca":0,"thal":1,"ca_missing":0}'
HEALTHY='{"age":35,"sex":0,"cp":0,"trestbps":120,"chol":180,"fbs":0,"restecg":1,"thalach":175,"exang":0,"oldpeak":0.2,"slope":2,"ca":0,"thal":2,"ca_missing":0}'

echo "Targeting ${API_URL}"
echo "Open Grafana dashboard 'Heart API' first - http://localhost:3000"
echo "Set time range to 'Last 5 minutes', auto-refresh '5s'."
echo

echo "[1/4] 30 disease-risk + 10 low-risk predictions (panels 4, 5)..."
for i in $(seq 1 30); do
    curl -s -o /dev/null -X POST "${API_URL}/predict" \
        -H 'Content-Type: application/json' -d "${DISEASE}"
done
for i in $(seq 1 10); do
    curl -s -o /dev/null -X POST "${API_URL}/predict" \
        -H 'Content-Type: application/json' -d "${HEALTHY}"
done

echo "[2/4] 50 health checks + 50 metadata calls (panel 1)..."
for i in $(seq 1 50); do
    curl -s -o /dev/null "${API_URL}/health"
    curl -s -o /dev/null "${API_URL}/metadata"
done

echo "[3/4] 20 malformed + 20 missing-key + 20 unknown-route requests (panels 3, 6)..."
for i in $(seq 1 20); do
    curl -s -o /dev/null -X POST "${API_URL}/predict" -d 'not json'
    curl -s -o /dev/null -X POST "${API_URL}/predict" \
        -H 'Content-Type: application/json' -d '{"age":50}'
    curl -s -o /dev/null "${API_URL}/does-not-exist"
done

echo "[4/4] 5 batch predictions of 100 records each (panel 2 - latency)..."
BATCH=$(python3 -c "import json; print(json.dumps([json.loads('${DISEASE}') for _ in range(100)]))")
for i in $(seq 1 5); do
    curl -s -o /dev/null -X POST "${API_URL}/predict" \
        -H 'Content-Type: application/json' -d "${BATCH}"
done

echo
echo "Done. Wait ~15-30s for Prometheus to scrape, then refresh Grafana."
echo "Total predictions sent: 30 disease + 10 healthy + 5 * 100 batch = 540"
