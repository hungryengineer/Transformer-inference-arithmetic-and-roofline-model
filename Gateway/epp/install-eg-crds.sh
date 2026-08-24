#!/bin/bash
# Installs Envoy Gateway CRDs from the Helm chart subchart.
# Only needed if EG was first installed with --set crds.enabled=false.
# EG CRDs only install on `helm install`, not `helm upgrade`.

set -euo pipefail

EG_VERSION="${EG_VERSION:-v1.8.3}"
CHART_DIR=$(mktemp -d)

echo "Pulling envoy-gateway Helm chart ${EG_VERSION}..."
helm pull oci://docker.io/envoyproxy/gateway-helm \
  --version "${EG_VERSION}" \
  -d "${CHART_DIR}" \
  --untar

CRD_DIR="${CHART_DIR}/gateway-helm/charts/crds/crds/generated"

if [ ! -d "${CRD_DIR}" ]; then
  echo "ERROR: CRD directory not found at ${CRD_DIR}"
  rm -rf "${CHART_DIR}"
  exit 1
fi

echo "Applying EG CRDs..."
kubectl apply -f "${CRD_DIR}/gateway.envoyproxy.io_backends.yaml"
kubectl apply -f "${CRD_DIR}/gateway.envoyproxy.io_backendtrafficpolicies.yaml"
kubectl apply -f "${CRD_DIR}/gateway.envoyproxy.io_clienttrafficpolicies.yaml"
kubectl apply -f "${CRD_DIR}/gateway.envoyproxy.io_envoyextensionpolicies.yaml"
kubectl apply -f "${CRD_DIR}/gateway.envoyproxy.io_envoypatchpolicies.yaml"
kubectl apply -f "${CRD_DIR}/gateway.envoyproxy.io_httproutefilters.yaml"
kubectl apply -f "${CRD_DIR}/gateway.envoyproxy.io_securitypolicies.yaml"

echo "Done. Cleaning up..."
rm -rf "${CHART_DIR}"
echo "EG CRDs installed successfully."
