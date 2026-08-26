<<<<<<< HEAD
# Transformer-inference-arithmetic-and-roofline-model
=======
1. time slicing: 3 way GPU time-slicing--> need helm upgrade nvdp nvdp/nvidia-device-plugin as time slicing is defined in this chart
2. Check with http requests how they are distributed across the three replicas (round-robin fashion)
3. Install api gateway and envoy gateway controller in k8s cluster: few crds like http route, gateway will be created

<<<<<<< HEAD
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
=======
4.  Client
   ↓
 Envoy Gateway (Data Plane Proxy)  <-- Evaluates HTTPRoute
   ↓                               <-- Matches InferencePool
 EPP (vllm-epp-epp)                <-- Proxy pauses and asks EPP via gRPC: "Which pod?"
   ↓                               <-- EPP replies: "Use Pod vLLM-1's IP"
 vLLM-1 (Direct Pod IP)            <-- Proxy forwards directly to the Pod!


5. queue-> KEDA -> service
scales on custom metric like num_requests_waiting
>>>>>>> e69ab0d (day 3 KEDA finished)
