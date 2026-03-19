# View logs from kubernetes cluster

```bash
ssh -o ConnectTimeout=10 theo@theo-ThinkPad-L430 "kubectl logs -f deployment/robinhoodbot -n robinhoodbot"
```