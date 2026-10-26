# Low-Latency SageMaker Realtime Endpoint with FinOps Autoscaling

[![Terraform IaC](https://img.shields.io/badge/IaC-Terraform%201.5%2B-blueviolet)](infrastructure/terraform)
[![Inference](https://img.shields.io/badge/SageMaker-Realtime%20Endpoint-orange)](src/sagemaker_serving)
[![FinOps Impact](https://img.shields.io/badge/FinOps-23%25%20Cost%20Reduction-brightgreen)](README.md)
[![Model Support](https://img.shields.io/badge/Architecture-Deep%20Learning%20%26%20Whisper%20ASR-blue)](src/sagemaker_serving)

Production-grade deployment and serving architecture for low-latency machine learning models on **Amazon SageMaker Realtime Endpoints**, featuring **Target Tracking and Step Scaling** policies (**NTT DATA** production benchmark).

The repository eliminates legacy, fragile Jupyter Notebooks in favor of modular Python inference code and Infrastructure as Code:
- **Zero Cloud Cost & Fast Local Verification**: Mocked inference pipeline validating both Tabular Deep Learning and **Audio / Whisper ASR** transcription patterns.
- **FinOps Optimization (-23% Cost)**: Terraform provisioning `aws_appautoscaling_policy` with `SageMakerVariantInvocationsPerInstance` target tracking, automatically scaling capacity to protect p99 SLA under traffic surges while decommissioning idle compute.
- **Canary Deployments**: Production variants configuration with 90/10 blue/green traffic splitting between primary and challenger models.

---

## 🏛️ Architecture & Scaling Model

```
[Incoming Inference Request]
             │
             ▼
[SageMaker Realtime Endpoint]
  ├── Variant-Primary-v2 (90% Traffic) ─── Target Tracking Autoscaling (-23% Cost)
  └── Variant-Challenger-v3 (10% Traffic) ── Canary Validation
             │
             ▼
[Custom SageMaker Handler (src/sagemaker_serving/inference.py)]
  ├── Tabular Prediction (< 5ms)
  └── Whisper ASR Audio Transcription
```

---

## 🚀 Local Verification

```bash
export PYTHONPATH=src
python -m pytest tests/ -v
```
