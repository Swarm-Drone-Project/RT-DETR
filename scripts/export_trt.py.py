#!/usr/bin/env python3
"""
Convert RT-DETR PyTorch model (.pth) to FP16 TensorRT engine (.engine).

This script uses the repository's YAMLConfig and model definitions (the
`rtdetrv2_pytorch/src` package) to reconstruct the RT-DETR model, load a
checkpoint (supports EMA), export the deployable model to ONNX, and build a
TensorRT FP16 engine.

Usage:
    python3 convert_pth_to_engine_fp16.py \
        --pth-model /path/to/train_robo.pth \
        --config /path/to/config.yml \
        --output-engine /path/to/output.engine
"""

import argparse
import sys
from pathlib import Path

import torch
import tensorrt as trt


def build_model_from_config(cfg_path: str, checkpoint_path: str, device: str = "cpu"):
    """Build RT-DETR model using YAMLConfig from rtdetrv2_pytorch and load state_dict.

    Returns the deployable model (cfg.model.deploy()) ready for export.
    """
    repo_root = Path(__file__).resolve().parents[1]
    rtdetr_pkg = repo_root / "rtdetrv2_pytorch"
    if str(rtdetr_pkg) not in sys.path:
        sys.path.insert(0, str(rtdetr_pkg))

    # import YAMLConfig from the repo
    from src.core import YAMLConfig

    print(f"[*] Loading config: {cfg_path}")
    cfg = YAMLConfig(cfg_path)

    print(f"[*] Loading checkpoint: {checkpoint_path}")
    checkpoint = torch.load(checkpoint_path, map_location='cpu')

    # Prefer EMA module if present
    if 'ema' in checkpoint and isinstance(checkpoint['ema'], dict) and 'module' in checkpoint['ema']:
        state = checkpoint['ema']['module']
        print('[*] Using EMA state from checkpoint')
    elif 'model' in checkpoint:
        state = checkpoint['model']
        print('[*] Using state_dict from checkpoint["model"]')
    else:
        raise ValueError('Checkpoint does not contain model state or ema.module')

    model = cfg.model
    model.load_state_dict(state)

    # Create deploy model and move to device
    deploy_model = model.deploy()
    deploy_model.eval()
    if device == 'cuda' and torch.cuda.is_available():
        deploy_model.to('cuda')

    return deploy_model


def export_to_onnx(model: torch.nn.Module, output_onnx: str, input_shape: tuple = (1, 1, 640, 640), device: str = 'cpu'):
    """Export a torch.nn.Module to ONNX.

    The deploy model should accept a single `images` tensor input.
    """
    print(f"[*] Exporting model to ONNX: {output_onnx}")

    if device == 'cuda' and torch.cuda.is_available():
        dummy_input = torch.randn(input_shape, device='cuda')
        model.to('cuda')
    else:
        dummy_input = torch.randn(input_shape, device='cpu')

    # Use dynamic axes for batch and spatial dims
    dynamic_axes = {'images': {0: 'batch', 2: 'height', 3: 'width'}}

    torch.onnx.export(
        model,
        dummy_input,
        output_onnx,
        input_names=['images'],
        output_names=['outputs'],
        opset_version=16,
        do_constant_folding=True,
        dynamic_axes=dynamic_axes,
    )

    print(f"[✓] ONNX model saved: {output_onnx}")


def onnx_to_tensorrt_fp16(onnx_path: str, engine_path: str, max_batch_size: int = 1):
    """Convert an ONNX file to TensorRT FP16 engine using the Python API."""
    print(f"[*] Converting ONNX to TensorRT FP16 engine...")
    print(f"    ONNX: {onnx_path}")
    print(f"    Engine: {engine_path}")

    logger = trt.Logger(trt.Logger.INFO)
    trt.init_libnvinfer_plugins(logger, namespace="")

    builder = trt.Builder(logger)
    EXPLICIT_BATCH = 1 << int(trt.NetworkDefinitionCreationFlag.EXPLICIT_BATCH)
    network = builder.create_network(EXPLICIT_BATCH)
    parser = trt.OnnxParser(network, logger)

    print("[*] Parsing ONNX model...")
    with open(onnx_path, 'rb') as f:
        if not parser.parse(f.read()):
            print('[!] ONNX parsing failed:')
            for i in range(parser.num_errors):
                print('   ', parser.get_error(i))
            raise RuntimeError('ONNX parsing failed')

    print('[✓] ONNX parsed successfully')

    config = builder.create_builder_config()
    config.set_memory_pool_limit(trt.MemoryPoolType.WORKSPACE, 1 << 30)  # 1GB

    # Create and add an optimization profile for dynamic input shapes
    profile = builder.create_optimization_profile()
    # Assume input 0 is the image tensor with shape [batch, channels, height, width]
    input_name = network.get_input(0).name
    # Conservative shapes: min 320, opt 640, max 1024 for spatial dims
    min_shape = (1, 1, 320, 320)
    opt_shape = (1, 1, 640, 640)
    max_shape = (1, 1, 1024, 1024)
    try:
        profile.set_shape(input_name, min=min_shape, opt=opt_shape, max=max_shape)
    except Exception:
        # Fallback ordering if trt expects tuples of ints
        profile.set_shape(input_name, min_shape, opt_shape, max_shape)
    config.add_optimization_profile(profile)

    if builder.platform_has_fast_fp16:
        print('[*] Enabling FP16 precision')
        config.set_flag(trt.BuilderFlag.FP16)
    else:
        print('[!] Platform does not have fast FP16 support; FP16 flag will still be set')
        config.set_flag(trt.BuilderFlag.FP16)

    print('[*] Building serialized engine... (this may take a while)')
    serialized_engine = builder.build_serialized_network(network, config)
    if serialized_engine is None:
        raise RuntimeError('Failed to build TensorRT engine')

    with open(engine_path, 'wb') as f:
        f.write(serialized_engine)

    size_mb = Path(engine_path).stat().st_size / (1024 * 1024)
    print(f"[✓] Engine saved: {engine_path} ({size_mb:.1f} MB)")


def main():
    parser = argparse.ArgumentParser(description='Convert RT-DETR PyTorch model to TensorRT FP16 engine')
    parser.add_argument('--pth-model', type=str, required=True)
    parser.add_argument('--config', type=str, required=True, help='YAML config file to build model')
    parser.add_argument('--output-onnx', type=str, default=None)
    parser.add_argument('--output-engine', type=str, required=True)
    parser.add_argument('--input-shape', type=int, nargs=4, default=[1, 1, 640, 640])
    parser.add_argument('--device', type=str, choices=['cuda', 'cpu'], default='cuda')
    parser.add_argument('--skip-onnx', action='store_true')
    args = parser.parse_args()

    pth_path = Path(args.pth_model)
    if not pth_path.exists():
        print(f'[!] Checkpoint not found: {pth_path}')
        raise SystemExit(1)

    if args.output_onnx is None:
        args.output_onnx = str(pth_path.parent / f"{pth_path.stem}.onnx")

    onnx_path = Path(args.output_onnx)
    engine_path = Path(args.output_engine)

    print('=' * 60)
    print('RT-DETR PyTorch -> ONNX -> TensorRT FP16')
    print('=' * 60)
    print(f'Checkpoint: {pth_path}')
    print(f'Config:     {args.config}')
    print(f'ONNX:       {onnx_path}')
    print(f'Engine:     {engine_path}')
    print(f'Input:      {tuple(args.input_shape)}')
    print(f'Device:     {args.device}')
    print('=' * 60)

    device = args.device
    if device == 'cuda' and not torch.cuda.is_available():
        print('[!] CUDA not available, falling back to CPU')
        device = 'cpu'

    try:
        model = build_model_from_config(args.config, str(pth_path), device=device)

        if onnx_path.exists() and args.skip_onnx:
            print(f'[*] Skipping ONNX export (exists): {onnx_path}')
        else:
            export_to_onnx(model, str(onnx_path), tuple(args.input_shape), device=device)

        onnx_to_tensorrt_fp16(str(onnx_path), str(engine_path), max_batch_size=1)

        print('[✓] Conversion complete')
    except Exception as e:
        print('[!] Conversion failed:', e)
        raise


if __name__ == '__main__':
    main()
