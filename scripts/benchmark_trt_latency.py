import os
import argparse
import numpy as np
import torch
import tensorrt as trt

def benchmark_trt(args):
    # Initialize TensorRT logger
    TRT_LOGGER = trt.Logger(trt.Logger.WARNING)
    trt.init_libnvinfer_plugins(TRT_LOGGER, "")
    
    print(f"Loading TensorRT engine from {args.engine}...")
    with open(args.engine, "rb") as f, trt.Runtime(TRT_LOGGER) as runtime:
        engine = runtime.deserialize_cuda_engine(f.read())
        
    if not engine:
        raise RuntimeError("Failed to load TensorRT engine.")
        
    context = engine.create_execution_context()
    
    # Check for dynamic shapes and set explicit shapes
    for i in range(engine.num_bindings):
        if engine.binding_is_input(i):
            shape = engine.get_binding_shape(i)
            # If batch size is dynamic (-1), set it to the requested batch size
            if -1 in shape:
                new_shape = tuple([args.batch_size if s == -1 else s for s in shape])
                context.set_binding_shape(i, new_shape)

    # Allocate memory using PyTorch tensors directly on the GPU
    # This avoids pycuda dependency and strictly isolates GPU execution time
    inputs = []
    outputs = []
    bindings = []
    
    print("Allocating GPU memory buffers...")
    for i in range(engine.num_bindings):
        shape = context.get_binding_shape(i)
        dtype = trt.nptype(engine.get_binding_dtype(i))
        
        # Map TensorRT dtype to PyTorch dtype
        if dtype == np.float32:
            torch_dtype = torch.float32
        elif dtype == np.float16:
            torch_dtype = torch.float16
        elif dtype == np.int32:
            torch_dtype = torch.int32
        elif dtype == np.int64:
            torch_dtype = torch.int64
        else:
            raise TypeError(f"Unsupported dtype: {dtype}")
            
        tensor = torch.empty(tuple(shape), dtype=torch_dtype, device='cuda')
        bindings.append(tensor.data_ptr())
        
        if engine.binding_is_input(i):
            # Fill inputs with dummy data
            if torch_dtype in [torch.float32, torch.float16]:
                tensor.normal_()
            else:
                tensor.zero_()
            inputs.append(tensor)
        else:
            outputs.append(tensor)
            
    stream = torch.cuda.Stream()
    
    # Warmup
    print(f"Warming up for {args.warmup} iterations...")
    for _ in range(args.warmup):
        context.execute_async_v2(bindings=bindings, stream_handle=stream.cuda_stream)
    stream.synchronize()
        
    # Benchmark
    print(f"Benchmarking for {args.iterations} iterations...")
    start_events = [torch.cuda.Event(enable_timing=True) for _ in range(args.iterations)]
    end_events = [torch.cuda.Event(enable_timing=True) for _ in range(args.iterations)]
    
    for i in range(args.iterations):
        start_events[i].record(stream)
        context.execute_async_v2(bindings=bindings, stream_handle=stream.cuda_stream)
        end_events[i].record(stream)
        
    stream.synchronize()
    
    # Calculate times
    times = [s.elapsed_time(e) for s, e in zip(start_events, end_events)]
    
    avg_time = sum(times) / len(times)
    min_time = min(times)
    max_time = max(times)
    fps = 1000.0 / avg_time * args.batch_size

    print("\n" + "="*40)
    print("      TENSORRT BENCHMARK RESULTS")
    print("="*40)
    print(f"Engine:          {args.engine}")
    print(f"Batch Size:      {args.batch_size}")
    print(f"Average Latency: {avg_time:.2f} ms")
    print(f"Min Latency:     {min_time:.2f} ms")
    print(f"Max Latency:     {max_time:.2f} ms")
    print(f"Estimated FPS:   {fps:.2f}")
    print("="*40)

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Benchmark RT-DETR TensorRT inference latency")
    parser.add_argument('-e', '--engine', type=str, required=True, help='Path to TensorRT engine file (.engine)')
    parser.add_argument('--batch_size', type=int, default=1, help='Batch size for benchmarking')
    parser.add_argument('--warmup', type=int, default=50, help='Number of warmup iterations')
    parser.add_argument('--iterations', type=int, default=200, help='Number of iterations for benchmarking')
    args = parser.parse_args()
    
    benchmark_trt(args)
