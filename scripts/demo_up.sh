#!/usr/bin/env bash
# One-shot bring-up of the Heart Disease API on a local kind cluster.
#
# This is the script referenced by the Task-7 Production Deployment
# instructions and by the Deliverable-(c) "access instructions for
# local testing" pointer. After it succeeds, the API is reachable on:
#
#     http://localhost/health
#     http://localhost/predict   (POST, JSON body)
#     http://localhost/metrics
#
# Prerequisites (one-time install):
#   - Docker (Engine or Desktop) running
#   - kind            (brew install kind)
#   - kubectl         (brew install kubectl)
#
# Usage:
#   bash scripts/demo_up.sh                # build + bring up + smoke test
#   bash scripts/demo_up.sh --skip-build   # skip docker build (image already exists)
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

CLUSTER_NAME="heart"
IMAGE="heart-api:latest"
SKIP_BUILD=0
[[ "${1:-}" == "--skip-build" ]] && SKIP_BUILD=1

step() { echo; echo "==> $*"; }
need() { command -v "$1" >/dev/null 2>&1 || { echo "missing required tool: $1" >&2; exit 1; }; }

need docker
need kind
need kubectl

step "1/6  Build the Docker image ($IMAGE)"
if [[ $SKIP_BUILD -eq 1 ]]; then
    echo "    --skip-build set, reusing existing image"
    docker image inspect "$IMAGE" >/dev/null
else
    docker build -f docker/Dockerfile -t "$IMAGE" .
fi
docker images "$IMAGE" --format 'table {{.Repository}}:{{.Tag}}\t{{.Size}}'

step "2/6  Create kind cluster '$CLUSTER_NAME' (if not present)"
if kind get clusters | grep -qx "$CLUSTER_NAME"; then
    echo "    cluster already exists; reusing it"
else
    kind create cluster --config k8s/setup/kind-cluster-config.yaml
fi
kubectl config use-context "kind-$CLUSTER_NAME" >/dev/null

step "3/6  Sideload image into the kind node"
kind load docker-image "$IMAGE" --name "$CLUSTER_NAME"

step "4/6  Install ingress-nginx (if not present)"
if kubectl get ns ingress-nginx >/dev/null 2>&1; then
    echo "    ingress-nginx namespace already exists; skipping install"
else
    kubectl apply -f https://raw.githubusercontent.com/kubernetes/ingress-nginx/main/deploy/static/provider/kind/deploy.yaml
fi
echo "    waiting for controller to be ready..."
kubectl wait --namespace ingress-nginx \
    --for=condition=ready pod \
    --selector=app.kubernetes.io/component=controller \
    --timeout=180s

step "5/6  Apply heart-api manifests"
kubectl apply -f k8s/configmap.yaml \
              -f k8s/deployment.yaml \
              -f k8s/service.yaml \
              -f k8s/hpa.yaml \
              -f k8s/ingress.yaml
kubectl rollout status deployment/heart-api --timeout=180s
kubectl get pods,svc,ingress,hpa -l app=heart-api

step "6/6  Smoke test via Ingress (http://localhost)"
sleep 3
HEALTH_CODE=$(curl -s -o /tmp/heart_health.json -w "%{http_code}" http://localhost/health)
echo "    GET  /health  → HTTP $HEALTH_CODE"
cat /tmp/heart_health.json; echo

PREDICT_CODE=$(curl -s -o /tmp/heart_predict.json -w "%{http_code}" \
    -X POST http://localhost/predict \
    -H 'Content-Type: application/json' \
    -d '{"age":63,"sex":1,"cp":3,"trestbps":145,"chol":233,"fbs":1,
         "restecg":0,"thalach":150,"exang":0,"oldpeak":2.3,"slope":0,
         "ca":0,"thal":1,"ca_missing":0}')
echo "    POST /predict → HTTP $PREDICT_CODE"
cat /tmp/heart_predict.json; echo

if [[ "$HEALTH_CODE" != "200" || "$PREDICT_CODE" != "200" ]]; then
    echo
    echo "❌ smoke test FAILED — see kubectl logs deploy/heart-api"
    exit 1
fi

echo
echo "✅ heart-api is up. Tear down with: bash scripts/demo_down.sh"
