# ADR 001: No Local CUDA

## Status

Accepted

## Context

Development machine has Intel UHD Graphics only. No NVIDIA GPU or `nvidia-smi` available.

## Decision

- Do not install CUDA toolkit locally.
- Use CPU for local inference development (MediaPipe, YOLOv8n).
- Use Google Colab or Kaggle for GPU-accelerated training.

## Consequences

- Training runs locally will be slow; plan cloud GPU sessions for YOLO fine-tuning.
- PyTorch CPU wheels are sufficient for development and demo.
