# Dev Environment Setup (RobinhoodBot)

## Build the image
```bash
docker build -t robinhoodbot-dev-sandbox:latest -f k8s/dev-environment/Dockerfile.dev robinhoodbot/
```

## Deploy to Docker (standalone, no Kubernetes)
```bash
docker volume create robinhoodbot-workspace
docker run -d --name robinhoodbot-dev -v robinhoodbot-workspace:/workspace -v robinhoodbot-home:/home/agent robinhoodbot-dev-sandbox:latest sleep infinity
docker exec -u root robinhoodbot-dev chown -R 1000:1000 /workspace
```

## Deploy to bare-metal Kubernetes

### Build and push to the cluster registry
Replace `REGISTRY_HOST` with your cluster's registry IP.

```bash
docker build -t REGISTRY_HOST:5000/robinhoodbot-dev-sandbox:latest -f k8s/dev-environment/Dockerfile.dev robinhoodbot/
docker push REGISTRY_HOST:5000/robinhoodbot-dev-sandbox:latest
```

### Apply the deployment
```bash
kubectl apply -f k8s/dev-environment/dev-pvc-metal.yaml -f k8s/dev-environment/dev-environment.metal.yaml
```

### Rebuild and redeploy (after Dockerfile changes)
```bash
docker build -t REGISTRY_HOST:5000/robinhoodbot-dev-sandbox:latest -f k8s/dev-environment/Dockerfile.dev robinhoodbot/
docker push REGISTRY_HOST:5000/robinhoodbot-dev-sandbox:latest
kubectl -n robinhoodbot-dev rollout restart deployment/dev-sandbox-robinhoodbot
```

### Connect to the dev container
```bash
kubectl -n robinhoodbot-dev exec -it deployment/dev-sandbox-robinhoodbot -- bash
```

### Clone the repo into the workspace
```bash
cd /workspace
git clone https://github.com/txavier/RobinhoodBot.git
cd RobinhoodBot
```
