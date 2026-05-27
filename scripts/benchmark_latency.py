import os
import sys
import time
import argparse
import torch

# Add the parent directory to sys.path to allow importing src
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '../rtdetrv2_pytorch'))

from src.core import YAMLConfig, yaml_utils

def benchmark(args):
    device = torch.device(args.device)
    
    # Load config
    update_dict = yaml_utils.parse_cli(args.update) if args.update else {}
    update_dict.update({k: v for k, v in args.__dict__.items() if k not in ['update'] and v is not None})
    print(f"Loading config from {args.config}")
    cfg = YAMLConfig(args.config, **update_dict)
    
    # Load model
    print("Deploying model (isolated from pre- and post-processing)...")
    model = cfg.model.deploy()
    
    # Load weights
    if args.weights:
        print(f"Loading weights from {args.weights}")
        checkpoint = torch.load(args.weights, map_location='cpu')
        if 'ema' in checkpoint:
            state = checkpoint['ema']['module']
        elif 'model' in checkpoint:
            state = checkpoint['model']
        else:
            state = checkpoint
        model.load_state_dict(state)
    else:
        print("No weights provided, benchmarking with randomly initialized weights.")
        
    model = model.to(device)
    model.eval()

    # Disable gradients for inference
    torch.set_grad_enabled(False)

    # Dummy input
    print(f"Generating dummy input with shape ({args.batch_size}, 3, {args.input_size}, {args.input_size})")
    x = torch.randn(args.batch_size, 3, args.input_size, args.input_size).to(device)
    
    if args.fp16 and device.type == 'cuda':
        print("Converting model and input to FP16...")
        model = model.half()
        x = x.half()

    # Warmup
    print(f"Warming up for {args.warmup} iterations...")
    for _ in range(args.warmup):
        _ = model(x)
        if device.type == 'cuda':
            torch.cuda.synchronize()

    # Benchmark
    print(f"Benchmarking for {args.iterations} iterations (isolated from pre/post-processing)...")
    if device.type == 'cuda':
        start_events = [torch.cuda.Event(enable_timing=True) for _ in range(args.iterations)]
        end_events = [torch.cuda.Event(enable_timing=True) for _ in range(args.iterations)]

    times = []
    
    for i in range(args.iterations):
        if device.type == 'cuda':
            start_events[i].record()
            _ = model(x)
            end_events[i].record()
        else:
            t0 = time.time()
            _ = model(x)
            t1 = time.time()
            times.append((t1 - t0) * 1000.0) # convert to ms

    if device.type == 'cuda':
        torch.cuda.synchronize()
        times = [s.elapsed_time(e) for s, e in zip(start_events, end_events)]
    
    # Calculate statistics
    avg_time = sum(times) / len(times)
    min_time = min(times)
    max_time = max(times)
    fps = 1000.0 / avg_time * args.batch_size
    
    print("\n" + "="*40)
    print("        BENCHMARK RESULTS")
    print("="*40)
    print(f"Device:          {device.type.upper()}{' (FP16)' if args.fp16 else ''}")
    print(f"Batch Size:      {args.batch_size}")
    print(f"Input Size:      {args.input_size}x{args.input_size}")
    print(f"Average Latency: {avg_time:.2f} ms")
    print(f"Min Latency:     {min_time:.2f} ms")
    print(f"Max Latency:     {max_time:.2f} ms")
    print(f"Estimated FPS:   {fps:.2f}")
    print("="*40)
    
if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Benchmark RT-DETR inference latency isolated from pre/post processing")
    parser.add_argument('-c', '--config', type=str, required=True, help='Path to config file')
    parser.add_argument('-w', '--weights', type=str, default=None, help='Path to weights (.pth) file')
    parser.add_argument('-d', '--device', type=str, default='cuda', help='Device to run benchmark on (e.g., cuda or cpu)')
    parser.add_argument('--input_size', type=int, default=640, help='Input size for benchmarking')
    parser.add_argument('--batch_size', type=int, default=1, help='Batch size for benchmarking')
    parser.add_argument('--warmup', type=int, default=50, help='Number of warmup iterations')
    parser.add_argument('--iterations', type=int, default=200, help='Number of iterations for benchmarking')
    parser.add_argument('--fp16', action='store_true', help='Use FP16 precision (only for CUDA)')
    parser.add_argument('-u', '--update', nargs='+', help='Update yaml config')
    args = parser.parse_args()
    
    benchmark(args)
