# PowerShell script for setting up local Kind cluster
Write-Host "Starting Kind local Kubernetes cluster (preface-dbn)..." -ForegroundColor Green

if (Get-Command kind -ErrorAction SilentlyContinue) {
    if (kind get clusters | Select-String -Pattern "preface-dbn") {
        Write-Host "Kind cluster 'preface-dbn' already exists. Skipping creation." -ForegroundColor Yellow
    } else {
        kind create cluster --name preface-dbn
    }
} else {
    Write-Host "kind command not found. Assuming cluster is already running or managed externally." -ForegroundColor Yellow
}

Write-Host "Cluster started successfully! Verifying cluster info..." -ForegroundColor Green
kubectl cluster-info
kubectl get nodes

Write-Host "Installing metrics-server for HPA..." -ForegroundColor Green
kubectl apply -f https://github.com/kubernetes-sigs/metrics-server/releases/latest/download/components.yaml

# Wait for deployment to be created
Start-Sleep -Seconds 5

$args = (kubectl get deployment metrics-server -n kube-system -o jsonpath='{.spec.template.spec.containers[0].args}') 2>$null
if ($args -and ($args -notmatch "--kubelet-insecure-tls")) {
    Write-Host "Patching metrics-server to allow insecure TLS (required for Kind)..."
    kubectl patch -n kube-system deployment metrics-server --type=json -p='[{\"op\": \"add\", \"path\": \"/spec/template/spec/containers/0/args/-\", \"value\": \"--kubelet-insecure-tls\"}]'
} else {
    Write-Host "metrics-server already patched or created." -ForegroundColor Yellow
}

Write-Host "Local cluster setup complete!" -ForegroundColor Green
