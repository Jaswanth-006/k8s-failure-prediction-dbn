#!/usr/bin/env bash
# Prepare Phase 3 Chaos Mesh Faults (Safe/Dry-Run Mode)

echo "Creating manifests/chaos directory..."
mkdir -p manifests/chaos

echo "Generating CPU Stress Fault manifest (manifests/chaos/cpu-stress-train.yaml)..."
cat <<'EOF' > manifests/chaos/cpu-stress-train.yaml
apiVersion: chaos-mesh.org/v1alpha1
kind: StressChaos
metadata:
  name: cpu-stress-train
  namespace: default
spec:
  mode: one
  selector:
    namespaces:
      - default
    labelSelectors:
      'app': 'ts-train-service'
  stressors:
    cpu:
      workers: 2
      load: 80
  duration: '10m'
EOF

echo "Generating Network Latency Fault manifest (manifests/chaos/network-delay-station.yaml)..."
cat <<'EOF' > manifests/chaos/network-delay-station.yaml
apiVersion: chaos-mesh.org/v1alpha1
kind: NetworkChaos
metadata:
  name: network-delay-station
  namespace: default
spec:
  action: delay
  mode: one
  selector:
    namespaces:
      - default
    labelSelectors:
      'app': 'ts-station-service'
  delay:
    latency: '300ms'
    jitter: '50ms'
  duration: '10m'
EOF

echo "Validating syntax with --dry-run=client..."
kubectl apply -f manifests/chaos/cpu-stress-train.yaml --dry-run=client
kubectl apply -f manifests/chaos/network-delay-station.yaml --dry-run=client

echo ""
echo "Fault injection definitions are prepared but NOT applied."
echo "Review the manifests in 'manifests/chaos/'."
echo "To execute fault injection, remove --dry-run=client from the commands above, or manually run:"
echo "  kubectl apply -f manifests/chaos/"
