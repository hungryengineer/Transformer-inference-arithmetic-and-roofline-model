#!/usr/bin/env bash
set -euo pipefail

GW="${GW:-http://192.168.1.40:8000}"
MODEL="${MODEL:-Qwen/Qwen2.5-0.5B-Instruct}"
WARMUP="${WARMUP:-10}"
N="${N:-30}"

# Long shared prefix (forces prefix-cache scorer to matter)
PREFIX="SHARED_KV_PREFIX_$(python3 - <<'PY'
print(("The quick brown fox jumps over the lazy dog. ") * 40)
PY
)"

count_pod() {
  local pod=$1
  kubectl exec "$pod" -- curl -s localhost:8000/metrics \
    | awk '/^vllm:request_success_total\{/ {s+=$NF} END{print s+0}'
}

echo "=== Baseline counters ==="
c0_before=$(count_pod vllm-0)
c1_before=$(count_pod vllm-1)
echo "vllm-0: $c0_before"
echo "vllm-1: $c1_before"

send() {
  local content=$1
  curl -sS --max-time 60 "$GW/v1/chat/completions" \
    -H 'Content-Type: application/json' \
    -d "$(python3 - <<PY
import json
print(json.dumps({
  "model": "$MODEL",
  "messages": [{"role": "user", "content": """$content"""}],
  "max_tokens": 5
}))
PY
)" >/dev/null
}

echo "=== Warmup ($WARMUP reqs, shared prefix) ==="
for i in $(seq 1 "$WARMUP"); do
  send "${PREFIX} warmup ${i}"
  printf '.'
done
echo

echo "=== Measure ($N reqs, same prefix) ==="
for i in $(seq 1 "$N"); do
  send "${PREFIX} ask ${i}"
  printf '.'
done
echo

echo "=== After counters ==="
c0_after=$(count_pod vllm-0)
c1_after=$(count_pod vllm-1)
d0=$((c0_after - c0_before))
d1=$((c1_after - c1_before))
total=$((d0 + d1))

echo "vllm-0 delta: $d0"
echo "vllm-1 delta: $d1"
echo "total delta:  $total"

if [ "$total" -eq 0 ]; then
  echo "No completed requests counted — check gateway/pods."
  exit 1
fi

# Rough skew: max share of traffic
if [ "$d0" -ge "$d1" ]; then
  maj=$d0; min=$d1; maj_name=vllm-0
else
  maj=$d1; min=$d0; maj_name=vllm-1
fi
pct=$(( maj * 100 / total ))
echo "majority: $maj_name ($pct%)"

if [ "$pct" -ge 70 ]; then
  echo "RESULT: skewed toward $maj_name → looks AI/prefix-aware"
else
  echo "RESULT: roughly even → looks closer to dumb LB"
fi