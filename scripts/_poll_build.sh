#!/bin/bash
for i in $(seq 1 240); do
  if pgrep -f "docker build -f docker/Dockerfile" >/dev/null; then
    sleep 5
  else
    echo "BUILD_FINISHED at iter=$i"
    break
  fi
done
echo "===== last 20 log lines ====="
tail -20 /tmp/dock_build.log
echo "===== image info ====="
/usr/local/bin/docker images heart-api --format '{{.Repository}}:{{.Tag}} {{.ID}} {{.CreatedSince}} {{.Size}}'
