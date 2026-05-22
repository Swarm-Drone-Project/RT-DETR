import argparse
import ctypes
import math
import os
import sys
import time
import queue as stdlib_queue
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import cv2
import h5py
import numpy as np
import tensorrt as trt
import gi
gi.require_version('Gst', '1.0')
from gi.repository import Gst

# --- CUDA Utility Class ---
class CudaRuntime:
    def __init__(self) -> None:
        from ctypes.util import find_library
        lib_path = find_library("cudart") or "libcudart.so"
        self.lib = ctypes.CDLL(lib_path)
        # Setup argument types for safety
        self.lib.cudaMalloc.argtypes = [ctypes.POINTER(ctypes.c_void_p), ctypes.c_size_t]
        self.lib.cudaMemcpyAsync.argtypes = [ctypes.c_void_p, ctypes.c_void_p, ctypes.c_size_t, ctypes.c_int, ctypes.c_void_p]
        self.lib.cudaStreamCreate.argtypes = [ctypes.POINTER(ctypes.c_void_p)]

    def malloc(self, num_bytes: int) -> int:
        ptr = ctypes.c_void_p()
        self.lib.cudaMalloc(ctypes.byref(ptr), num_bytes)
        return int(ptr.value)

    def free(self, ptr: int) -> None:
        if ptr: self.lib.cudaFree(ctypes.c_void_p(ptr))

    def memcpy_htod_async(self, dst: int, src: np.ndarray, num_bytes: int, stream: int) -> None:
        self.lib.cudaMemcpyAsync(ctypes.c_void_p(dst), ctypes.c_void_p(src.ctypes.data), num_bytes, 1, ctypes.c_void_p(stream))

    def memcpy_dtoh_async(self, dst: np.ndarray, src: int, num_bytes: int, stream: int) -> None:
        self.lib.cudaMemcpyAsync(ctypes.c_void_p(dst.ctypes.data), ctypes.c_void_p(src), num_bytes, 2, ctypes.c_void_p(stream))

    def create_stream(self) -> int:
        stream = ctypes.c_void_p()
        self.lib.cudaStreamCreate(ctypes.byref(stream))
        return int(stream.value)

    def synchronize_stream(self, stream: int) -> None:
        self.lib.cudaStreamSynchronize(ctypes.c_void_p(stream))

# --- HDF5 Logging ---
class HDF5ImageLogger:
    def __init__(self, output_dir):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.h5_path = self.output_dir / "annotated_frames.h5"
        self.h5_file = h5py.File(self.h5_path, "w")
        self.frames_group = self.h5_file.create_group("frames")
        self.frame_count = 0

    def append(self, frame, offset_s, timestamp_us):
        ds_name = f"frame_{self.frame_count:06d}"
        ds = self.frames_group.create_dataset(ds_name, data=frame, compression="lzf")
        ds.attrs["offset_s"] = offset_s
        ds.attrs["timestamp_us"] = timestamp_us
        self.frame_count += 1

    def close(self):
        if hasattr(self, "h5_file"): self.h5_file.close()

# --- GStreamer Frame Queue ---
frame_queue = stdlib_queue.Queue(maxsize=4)

def on_new_sample(appsink):
    sample = appsink.emit("pull-sample")
    if not sample: return Gst.FlowReturn.ERROR
    buf = sample.get_buffer()
    t_frame = int(time.time() * 1e6)
    ok, map_info = buf.map(Gst.MapFlags.READ)
    if ok:
        frame = np.frombuffer(map_info.data, dtype=np.uint8).reshape((640, 640, 3)).copy()
        buf.unmap(map_info)
        try:
            frame_queue.put_nowait((t_frame, frame))
        except stdlib_queue.Full:
            frame_queue.get_nowait(); frame_queue.put_nowait((t_frame, frame))
    return Gst.FlowReturn.OK

# --- Main Execution ---
def run(engine, conf_thres, output_dir):
    # 1. Initialize TRT Engine
    trt_logger = trt.Logger(trt.Logger.WARNING)
    trt.init_libnvinfer_plugins(trt_logger, "")
    with open(engine, "rb") as f:
        runtime = trt.Runtime(trt_logger)
        engine_obj = runtime.deserialize_cuda_engine(f.read())
    
    context = engine_obj.create_execution_context()
    cuda = CudaRuntime()
    stream = cuda.create_stream()

    # 2. Setup GPU Buffers
    input_name = engine_obj.get_tensor_name(0)
    output_names = [engine_obj.get_tensor_name(i) for i in range(1, engine_obj.num_io_tensors)]
    
    # Assume 640x640 FP32 for RT-DETR input
    buffers = {input_name: cuda.malloc(1 * 3 * 640 * 640 * 4)}
    for name in output_names:
        shape = context.get_tensor_shape(name)
        buffers[name] = cuda.malloc(np.prod(shape) * 4)

    # 3. GStreamer Pipeline
    Gst.init(None)
    pipeline_str = (f"v4l2src device=/dev/video0 ! video/x-raw, width=1600, height=1300, framerate=30/1 "
                    f"! videoscale ! video/x-raw, width=640, height=640 ! videoconvert ! video/x-raw, format=RGB "
                    f"! appsink name=sink emit-signals=true max-buffers=2 drop=true")
    pipeline = Gst.parse_launch(pipeline_str)
    pipeline.get_by_name("sink").connect("new-sample", on_new_sample)
    pipeline.set_state(Gst.State.PLAYING)

    h5_logger = HDF5ImageLogger(output_dir)
    start_perf = time.perf_counter()

    print(f"Starting RT-DETR: {engine} | Threshold: {conf_thres}")

    try:
        while True:
            t_frame, frame = frame_queue.get()
            
            # Preprocessing (Transpose HWC to CHW and Normalize)
            input_data = (frame.transpose(2, 0, 1).astype(np.float32) / 255.0).astype(np.float32)
            cuda.memcpy_htod_async(buffers[input_name], input_data, input_data.nbytes, stream)
            
            context.set_input_shape(input_name, (1, 3, 640, 640))
            for name, ptr in buffers.items():
                context.set_tensor_address(name, ptr)
            
            context.execute_async_v3(stream)
            
            # Retrieve Results
            outputs = {}
            for name in output_names:
                shape = context.get_tensor_shape(name)
                out_np = np.empty(shape, dtype=np.float32)
                cuda.memcpy_dtoh_async(out_np, buffers[name], out_np.nbytes, stream)
                outputs[name] = out_np
            
            cuda.synchronize_stream(stream)

            # --- Basic Filtering based on conf_thres ---
            # (Logic varies slightly by how your RT-DETR engine was exported)
            scores = outputs.get('scores', np.array([]))
            if scores.size > 0:
                valid_idx = np.where(scores > conf_thres)[0]
                if len(valid_idx) > 0:
                    print(f"Detected {len(valid_idx)} objects above {conf_thres}")

            # Visualization & Logging
            frame_bgr = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
            h5_logger.append(frame_bgr, time.perf_counter() - start_perf, t_frame)
            
            cv2.imshow("RT-DETR Drone Localisation", frame_bgr)
            if cv2.waitKey(1) & 0xFF == ord('q'): break

    finally:
        pipeline.set_state(Gst.State.NULL)
        h5_logger.close()
        for ptr in buffers.values(): cuda.free(ptr)
        print("Cleanup complete.")

def parse_opt():
    parser = argparse.ArgumentParser(description="RT-DETR Zero-Copy HDF5 Logger")
    parser.add_argument(
        "--engine", 
        type=str, 
        default="/home/jetsonnano/Documents/cv_pipeline_v1/RT-DETR/files/train_robo_r18_gray_fp16.engine", 
        help="Path to the TensorRT engine file"
    )
    parser.add_argument(
        "--conf-thres", 
        type=float, 
        default=0.5, 
        help="Confidence threshold for detections"
    )
    parser.add_argument(
        "--output-dir", 
        type=str, 
        default="/home/jetsonnano/Documents/cv_pipeline_v1/RT-DETR/files/log", 
        help="Directory to save HDF5 logs"
    )
    return parser.parse_args()

if __name__ == "__main__":
    opt = parse_opt()
    run(**vars(opt))