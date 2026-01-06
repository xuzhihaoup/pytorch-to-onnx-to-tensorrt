import tensorrt as trt
import pycuda.driver as cuda
import pycuda.autoinit  # 必须：初始化 CUDA context
import numpy as np
import cv2
import time

ENGINE_PATH = "srcnn_1.engine"
IMAGE_PATH = "face.png"
BATCH_SIZE = 1

# =====================================================
# 1. 加载 TensorRT Engine
# =====================================================
TRT_LOGGER = trt.Logger(trt.Logger.ERROR)

with open(ENGINE_PATH, "rb") as f, trt.Runtime(TRT_LOGGER) as runtime:
    engine = runtime.deserialize_cuda_engine(f.read())

context = engine.create_execution_context()

# =====================================================
# 2. 准备输入数据 (batch = 48)
# =====================================================
img = cv2.imread("face.png").astype(np.float32) 

# HWC -> CHW
img = np.transpose(img, (2, 0, 1))

# batch = 48
img = np.stack([img] * 1, axis=0)   # (48, 3, H, W)

# 强制 contiguous（非常重要）
img = np.ascontiguousarray(img)

context.set_input_shape("input", img.shape)

print("Input shape:", img.shape)
print("Contiguous:", img.flags['C_CONTIGUOUS'])

# =====================================================
# 3. 分配 GPU buffer（TensorRT 10 写法）
# =====================================================
stream = cuda.Stream()

# ---------- Input ----------
d_input = cuda.mem_alloc(int(img.nbytes))
context.set_tensor_address("input", int(d_input))

# ---------- Output ----------
output_shape = context.get_tensor_shape("output")
output_size = int(np.prod(output_shape)) * np.float32().nbytes

d_output = cuda.mem_alloc(output_size)
context.set_tensor_address("output", int(d_output))

h_output = np.empty(output_shape, dtype=np.float32)

print("Output shape:", output_shape)
print("Output bytes:", output_size)


# =====================================================
# 4. 推理
# =====================================================
cuda.memcpy_htod_async(d_input, img, stream)

# warmup
for _ in range(5):
    context.execute_async_v3(stream.handle)
stream.synchronize()

start = time.time()
context.execute_async_v3(stream.handle)
cuda.memcpy_dtoh_async(h_output, d_output, stream)
stream.synchronize()
end = time.time()

print(f"TensorRT inference time: {(end - start) * 1000:.2f} ms")

# =====================================================
# 5. 后处理（取 batch 中第 0 张）
# =====================================================
out = h_output[0]                 # (3, H, W)
out = np.transpose(out, (1, 2, 0))
out = np.clip(out, 0, 255).astype(np.uint8)

cv2.imwrite("face_trt.png", out)
print("Output saved as face_trt.png")
