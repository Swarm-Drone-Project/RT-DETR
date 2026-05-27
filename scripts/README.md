# Benchmark Scripts

This directory contains utility scripts to rigorously benchmark the inference latency of the RT-DETR model, primarily aimed at testing performance on edge devices such as the Jetson Nano.

Both scripts guarantee that they isolate the core network's forward pass, eliminating any external overhead introduced by data loading, pre-processing, or Non-Maximum Suppression (NMS) post-processing.

## 1. PyTorch Baseline Benchmark (`benchmark_latency.py`)

Benchmarks the standard PyTorch `.pth` model natively. It loads the deployed model format, skips gradients, and uses precise `torch.cuda.Event` timings.

### Usage:

Run this from the `rtdetrv2_pytorch` directory:
```bash
python ../scripts/benchmark_latency.py \
    --config configs/rtdetrv2/rtdetrv2_r18vd_120e_coco.yml \
    --weights weights/train_robo.pth \
    --device cuda \
    --fp16
```
*(Optionally change `--device cpu` or remove `--fp16` if testing on a non-CUDA device).*

## 2. TensorRT Benchmark (`benchmark_trt_latency.py`)

Benchmarks the compiled TensorRT `.engine` execution. 
- Uses PyTorch to manage GPU memory buffers without needing to compile `pycuda`.
- Memory transfers are deliberately kept out of the timing loop.
- Only the `execute_async_v2` call is profiled using native CUDA streams.

### Usage:

Run this from the `rtdetrv2_pytorch` directory:
```bash
python ../scripts/benchmark_trt_latency.py \
    -e model.engine \
    --batch_size 1
```

---

## Jetson Nano Recommended Workflow

To achieve the lowest latency and maximum frame-rate on the Jetson Nano, you must leverage TensorRT with FP16 enabled. 

**Step 1: Export ONNX model**
```bash
cd rtdetrv2_pytorch
python tools/export_onnx.py \
    -c configs/rtdetrv2/rtdetrv2_r18vd_120e_coco.yml \
    -r weights/train_robo.pth \
    -o model.onnx
```

**Step 2: Compile TensorRT Engine (FP16)**
```bash
python tools/export_trt.py \
    --onnx model.onnx \
    --saveEngine model.engine \
    --fp16
```

**Step 3: Benchmark**
```bash
python ../scripts/benchmark_trt_latency.py -e model.engine --batch_size 1
```
