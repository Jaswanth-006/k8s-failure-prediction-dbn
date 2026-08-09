#!/usr/bin/env bash
# Deploy Istio Minimal Profile and Prometheus Monitoring Stack (Phase 1 Lightweight)

echo "Checking for istioctl..."
if ! command -v istioctl &> /dev/null; then
    echo "ERROR: istioctl could not be found!"
    echo "Please download Istio (e.g., v1.22.1) and add istioctl to your PATH."
    echo "Windows users: download the .zip, extract, and add the bin directory to PATH."
    exit 1
fi

echo "Installing Istio Minimal Profile for Service Graph Telemetry..."
istioctl install --set profile=minimal -y

echo "Enabling Istio injection on default namespace (Phase 1 workload)..."
kubectl label namespace default istio-injection=enabled --overwrite

echo "Restarting Phase 1 microservices to inject Envoy sidecars (if already deployed)..."
kubectl rollout restart deployment ts-ui-dashboard ts-user-service ts-train-service ts-route-service ts-order-service ts-payment-service ts-inventory-service ts-station-service || true

echo "Deploying lightweight standalone Prometheus..."
kubectl create namespace monitoring || true
helm repo add prometheus-community https://prometheus-community.github.io/helm-charts
helm repo update

# Check if prometheus is already installed to prevent helm install from failing
if helm status prometheus -n monitoring > /dev/null 2>&1; then
    echo "Prometheus is already installed. Upgrading to ensure correct config..."
    HELM_CMD="upgrade"
else
    HELM_CMD="install"
fi

helm $HELM_CMD prometheus prometheus-community/prometheus \
  --namespace monitoring \
  --set alertmanager.enabled=false \
  --set prometheus-pushgateway.enabled=false \
  --set prometheus-node-exporter.enabled=false \
  --set kube-state-metrics.enabled=false \
  --set server.resources.requests.memory=256Mi \
  --set server.resources.limits.memory=1Gi

echo "Telemetry stack deployed successfully!"
