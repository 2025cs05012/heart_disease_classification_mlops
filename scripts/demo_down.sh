#!/usr/bin/env bash
# Tear down the local heart-api demo brought up by demo_up.sh.
#
# Default behaviour: remove just the heart-api workload, leave the
# kind cluster + ingress-nginx running (faster to recreate).
#
# Pass --full to also delete the kind cluster.
#
# Usage:
#   bash scripts/demo_down.sh           # remove heart-api workload only
#   bash scripts/demo_down.sh --full    # also delete the kind cluster
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

CLUSTER_NAME="heart"
FULL=0
[[ "${1:-}" == "--full" ]] && FULL=1

step() { echo; echo "==> $*"; }

step "Removing heart-api workload"
kubectl delete --ignore-not-found=true \
    -f k8s/ingress.yaml \
    -f k8s/hpa.yaml \
    -f k8s/service.yaml \
    -f k8s/deployment.yaml \
    -f k8s/configmap.yaml

if [[ $FULL -eq 1 ]]; then
    step "Deleting kind cluster '$CLUSTER_NAME'"
    kind delete cluster --name "$CLUSTER_NAME"
else
    echo
    echo "ℹ️  Cluster '$CLUSTER_NAME' and ingress-nginx left running."
    echo "   Re-run scripts/demo_up.sh --skip-build to bring the API back up."
    echo "   Pass --full to scripts/demo_down.sh to also delete the cluster."
fi
