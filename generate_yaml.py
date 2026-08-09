import yaml

app_code = """import os
import json
import urllib.request
from http.server import BaseHTTPRequestHandler, HTTPServer

ROUTES = os.getenv("ROUTES", "")
DOWNSTREAM = os.getenv("DOWNSTREAM", "")
SERVICE_NAME = os.getenv("SERVICE_NAME", "unknown")

class MockHandler(BaseHTTPRequestHandler):
    def do_GET(self): self.handle_req()
    def do_POST(self): self.handle_req()
        
    def handle_req(self):
        results = []
        if ROUTES:
            routes = dict(r.split('=') for r in ROUTES.split(',') if '=' in r)
            matched = False
            for prefix, target in routes.items():
                if self.path.startswith(prefix):
                    matched = True
                    try:
                        req = urllib.request.Request(f"http://{target}:8080{self.path}", method=self.command)
                        with urllib.request.urlopen(req, timeout=5) as response:
                            results.append({f"Proxied to {target}": json.loads(response.read().decode('utf-8'))})
                    except Exception as e:
                        results.append({f"Error {target}": str(e)})
                    break
            if not matched: results.append("No route matched")
        
        if DOWNSTREAM:
            for target in DOWNSTREAM.split(','):
                if target:
                    try:
                        req = urllib.request.Request(f"http://{target}:8080/", method="GET")
                        with urllib.request.urlopen(req, timeout=5) as response:
                            results.append({f"Called {target}": json.loads(response.read().decode('utf-8'))})
                    except Exception as e:
                        results.append({f"Error {target}": str(e)})

        # Dummy CPU load for HPA testing
        x = 0
        for _ in range(5000): x += 1

        self.send_response(200)
        self.send_header('Content-Type', 'application/json')
        self.end_headers()
        self.wfile.write(json.dumps({
            "status": "ok", 
            "service": SERVICE_NAME, 
            "path": self.path,
            "actions": results
        }).encode('utf-8'))

if __name__ == "__main__":
    print(f"Starting {SERVICE_NAME} on port 8080")
    server = HTTPServer(('0.0.0.0', 8080), MockHandler)
    server.serve_forever()
"""

services = {
    "ts-ui-dashboard": {"env": {"ROUTES": "/api/v1/trainservice=ts-train-service,/api/v1/userservice=ts-user-service,/api/v1/orderservice=ts-order-service,/api/v1/stationservice=ts-station-service"}, "hpa": True},
    "ts-user-service": {"env": {}, "hpa": False},
    "ts-train-service": {"env": {"DOWNSTREAM": "ts-route-service"}, "hpa": True},
    "ts-route-service": {"env": {}, "hpa": False},
    "ts-order-service": {"env": {"DOWNSTREAM": "ts-payment-service"}, "hpa": True},
    "ts-payment-service": {"env": {"DOWNSTREAM": "ts-inventory-service"}, "hpa": False},
    "ts-inventory-service": {"env": {}, "hpa": False},
    "ts-station-service": {"env": {}, "hpa": False}
}

manifests = []
manifests.append({
    "apiVersion": "v1",
    "kind": "ConfigMap",
    "metadata": {"name": "mock-app-code"},
    "data": {"app.py": app_code}
})

for svc, config in services.items():
    env_vars = [{"name": "SERVICE_NAME", "value": svc}]
    for k, v in config["env"].items():
        env_vars.append({"name": k, "value": v})
        
    manifests.append({
        "apiVersion": "apps/v1",
        "kind": "Deployment",
        "metadata": {"name": svc, "labels": {"app": svc}},
        "spec": {
            "replicas": 1,
            "selector": {"matchLabels": {"app": svc}},
            "template": {
                "metadata": {"labels": {"app": svc}},
                "spec": {
                    "containers": [{
                        "name": svc,
                        "image": "python:3.10-alpine",
                        "command": ["python", "-u", "/app/app.py"],
                        "ports": [{"containerPort": 8080}],
                        "env": env_vars,
                        "volumeMounts": [{"name": "code", "mountPath": "/app"}],
                        "resources": {
                            "requests": {"cpu": "10m", "memory": "32Mi"},
                            "limits": {"cpu": "150m", "memory": "128Mi"}
                        }
                    }],
                    "volumes": [{"name": "code", "configMap": {"name": "mock-app-code"}}]
                }
            }
        }
    })
    
    manifests.append({
        "apiVersion": "v1",
        "kind": "Service",
        "metadata": {"name": svc},
        "spec": {
            "selector": {"app": svc},
            "ports": [{"port": 8080, "targetPort": 8080}]
        }
    })
    
    if config["hpa"]:
        manifests.append({
            "apiVersion": "autoscaling/v2",
            "kind": "HorizontalPodAutoscaler",
            "metadata": {"name": f"{svc}-hpa"},
            "spec": {
                "scaleTargetRef": {
                    "apiVersion": "apps/v1",
                    "kind": "Deployment",
                    "name": svc
                },
                "minReplicas": 1,
                "maxReplicas": 3,
                "metrics": [{
                    "type": "Resource",
                    "resource": {
                        "name": "cpu",
                        "target": {
                            "type": "Utilization",
                            "averageUtilization": 50
                        }
                    }
                }]
            }
        })

with open('phase1-workload.yaml', 'w') as f:
    yaml.dump_all(manifests, f, sort_keys=False)
