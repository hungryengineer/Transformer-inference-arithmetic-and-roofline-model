<<<<<<< HEAD
# Transformer-inference-arithmetic-and-roofline-model
=======
1. time slicing: 3 way GPU time-slicing--> need helm upgrade nvdp nvdp/nvidia-device-plugin as time slicing is defined in this chart
2. Check with http requests how many they are distributed across the three replicas (round-robin fashion)
3. Install api gateway and envoy gateway controller in k8s cluster: few crds like http route, gateway will be created

4. Client
  ↓
Gateway API
  ↓
Envoy Gateway
  ↓
HTTPRoute
  ↓
vllm-svc
  ↓
vLLM-0 / vLLM-1
>>>>>>> 0870c09 (ai aware inference platform)
