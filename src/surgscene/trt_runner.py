"""Minimal TensorRT runner (Python) for parity and non-inferiority tests. Buffers are torch tensors."""

from pathlib import Path

import numpy as np
import tensorrt as trt
import torch

LOGGER = trt.Logger(trt.Logger.WARNING)
_DT = {trt.float32: torch.float32, trt.float16: torch.float16, trt.uint8: torch.uint8, trt.int32: torch.int32,
       trt.int64: torch.int64}


FP32_TYPES_VIT = ("NORMALIZATION", "SOFTMAX", "REDUCE")  # what torch autocast keeps in FP32 for a ViT


def build_engine(onnx_path: Path, engine_path: Path, fp16: bool, fp32_outside: str | None = None,
                 fp32_types: tuple = (), obey: bool = False) -> None:
    """fp32_outside: keep every float layer whose name doesn't start with this prefix (e.g. "/net/",
    the network submodule) in FP32. The decode graph turns a flat argmax index (up to 360k) into
    pixel coordinates; in FP16 that overflows (max 65504) and quantizes x near 1400 to 1 px.
    fp32_types: TensorRT layer types kept in FP32 everywhere (FP32_TYPES_VIT for a ViT: with LayerNorm
    and softmax in FP16, DINOv2 keypoints moved 0.8 px; torch autocast keeps these in FP32)."""
    builder = trt.Builder(LOGGER)
    network = builder.create_network(0)
    parser = trt.OnnxParser(network, LOGGER)
    if not parser.parse(onnx_path.read_bytes()):
        raise RuntimeError("\n".join(str(parser.get_error(i)) for i in range(parser.num_errors)))
    config = builder.create_builder_config()
    config.set_memory_pool_limit(trt.MemoryPoolType.WORKSPACE, 4 << 30)
    if fp16:
        config.set_flag(trt.BuilderFlag.FP16)
    if fp16 and (fp32_outside or fp32_types):
        config.set_flag(trt.BuilderFlag.OBEY_PRECISION_CONSTRAINTS if obey else trt.BuilderFlag.PREFER_PRECISION_CONSTRAINTS)
        floats = (trt.float32, trt.float16)
        pin = {getattr(trt.LayerType, t) for t in fp32_types}
        for i in range(network.num_layers):
            layer = network.get_layer(i)
            if layer.type in (trt.LayerType.CONSTANT, trt.LayerType.SHAPE):
                continue
            inside = fp32_outside is None or layer.name.startswith(fp32_outside)
            if inside and layer.type not in pin:
                continue
            if layer.num_outputs and all(layer.get_output(j).dtype in floats for j in range(layer.num_outputs)):
                layer.precision = trt.float32
                for j in range(layer.num_outputs):
                    layer.set_output_type(j, trt.float32)
    blob = builder.build_serialized_network(network, config)
    if blob is None:
        raise RuntimeError("engine build failed")
    engine_path.write_bytes(bytes(blob))


class TrtRunner:
    def __init__(self, engine_path: Path):
        self.engine = trt.Runtime(LOGGER).deserialize_cuda_engine(Path(engine_path).read_bytes())
        self.ctx = self.engine.create_execution_context()
        self.names = [self.engine.get_tensor_name(i) for i in range(self.engine.num_io_tensors)]
        self.bufs = {}
        for n in self.names:
            shape = tuple(self.engine.get_tensor_shape(n))
            self.bufs[n] = torch.empty(shape, dtype=_DT[self.engine.get_tensor_dtype(n)], device="cuda")
            self.ctx.set_tensor_address(n, self.bufs[n].data_ptr())
        self.inputs = [n for n in self.names if self.engine.get_tensor_mode(n) == trt.TensorIOMode.INPUT]
        self.outputs = [n for n in self.names if self.engine.get_tensor_mode(n) == trt.TensorIOMode.OUTPUT]
        self.stream = torch.cuda.Stream()

    def __call__(self, frames: np.ndarray) -> dict[str, np.ndarray]:
        self.bufs[self.inputs[0]].copy_(torch.from_numpy(frames))
        with torch.cuda.stream(self.stream):
            self.ctx.execute_async_v3(self.stream.cuda_stream)
        self.stream.synchronize()
        return {n: self.bufs[n].float().cpu().numpy() for n in self.outputs}
