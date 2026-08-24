#!/bin/bash
# Deploys the Gateway API Inference Extension (EPP) with vLLM.
# Prerequisites:
#   - k3s cluster running
#   - Envoy Gateway installed via Helm (v1.8.3+)
#   - vLLM StatefulSet running (vllm-0, vllm-1)
#
# Usage:
#   ./apply.sh          # apply all resources
#   ./apply.sh --delete # tear down all resources

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

if [ "${1:-}" = "--delete" ]; then
  echo "Tearing down EPP resources..."
  kubectl delete -f "${SCRIPT_DIR}/envoy-extension-policy.yaml" --ignore-not-found
  kubectl delete -f "${SCRIPT_DIR}/httproute.yaml" --ignore-not-found
  kubectl delete -f "${SCRIPT_DIR}/inference-pool.yaml" --ignore-not-found
  kubectl delete -f "${SCRIPT_DIR}/epp-deployment.yaml" --ignore-not-found
  kubectl delete -f "${SCRIPT_DIR}/epp-service-configmap.yaml" --ignore-not-found
  kubectl delete -f "${SCRIPT_DIR}/epp-rbac.yaml" --ignore-not-found
  echo "Done."
  exit 0
fi

echo "=== Step 1: Ensure EG CRDs are installed ==="
if ! kubectl get crd envoyextensionpolicies.gateway.envoyproxy.io &>/dev/null; then
  echo "EnvoyExtensionPolicy CRD not found. Installing EG CRDs..."
  bash "${SCRIPT_DIR}/install-eg-crds.sh"
else
  echo "EG CRDs already installed."
fi

echo ""
echo "=== Step 2: Apply RBAC ==="
kubectl apply -f "${SCRIPT_DIR}/epp-rbac.yaml"

echo ""
echo "=== Step 3: Apply ConfigMap and Service ==="
kubectl apply -f "${SCRIPT_DIR}/epp-service-configmap.yaml"

echo ""
echo "=== Step 4: Apply InferencePool ==="
kubectl apply -f "${SCRIPT_DIR}/inference-pool.yaml"

echo ""
echo "=== Step 5: Apply EPP Deployment ==="
kubectl apply -f "${SCRIPT_DIR}/epp-deployment.yaml"

echo ""
echo "=== Step 6: Apply HTTPRoute (InferencePool backend) ==="
kubectl apply -f "${SCRIPT_DIR}/httproute.yaml"

echo ""
echo "=== Step 7: Apply Backend (h2c) + EnvoyExtensionPolicy (FullDuplexStreamed) ==="
# EG v1.8.3 cannot use InferencePool as HTTPRoute backendRef (InvalidKind).
# AI-aware routing works via: HTTPRoute → vllm-svc + ExtProc → EPP scorers.
# Requires: enableBackend=true on EnvoyGateway, EPP --secure-serving=false, h2c Backend.
kubectl apply -f "${SCRIPT_DIR}/envoy-extension-policy.yaml"

echo ""
echo "=== All resources applied. Waiting for pods ==="
kubectl rollout status deployment/vllm-epp-epp --timeout=60s

echo ""
echo "=== Verifying ==="
echo "InferencePool:"
kubectl get inferencepool vllm-epp -o wide
echo ""
echo "EPP Pod:"
kubectl get pods -l llm-d-router-gateway=vllm-epp-epp
echo ""
echo "EnvoyExtensionPolicy:"
kubectl get envoyextensionpolicy vllm-epp-ext-proc
echo ""
echo "Done. Test with:"
echo "  curl -X POST http://<GATEWAY_IP>/v1/chat/completions \\"
echo "    -H 'Content-Type: application/json' \\"
echo "    -d '{\"model\":\"Qwen/Qwen2.5-0.5B-Instruct\",\"messages\":[{\"role\":\"user\",\"content\":\"Hello\"}],\"max_tokens\":10}'"
