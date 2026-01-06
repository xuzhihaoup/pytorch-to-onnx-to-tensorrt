import tensorrt as trt
import onnx

ONNX_PATH = "srcnn_1.onnx"
ENGINE_PATH = "srcnn_1.engine"

TRT_LOGGER = trt.Logger(trt.Logger.INFO)

# 1. Builder / Network
builder = trt.Builder(TRT_LOGGER)
EXPLICIT_BATCH = 1 << int(trt.NetworkDefinitionCreationFlag.EXPLICIT_BATCH)
network = builder.create_network(EXPLICIT_BATCH)

# 2. Parser
parser = trt.OnnxParser(network, TRT_LOGGER)
with open(ONNX_PATH, "rb") as f:
    if not parser.parse(f.read()):
        for i in range(parser.num_errors):
            print(parser.get_error(i))
        raise RuntimeError("ONNX parse failed")

# 3. Config
config = builder.create_builder_config()
config.set_memory_pool_limit(trt.MemoryPoolType.WORKSPACE, 1 << 30)

if builder.platform_has_fast_fp16:
    config.set_flag(trt.BuilderFlag.FP16)

# 4. Optimization profile（显式 batch 必须）
profile = builder.create_optimization_profile()
profile.set_shape(
    "input",
    min=(1, 3, 256, 256),
    opt=(1, 3, 256, 256),
    max=(1, 3, 256, 256),
)
config.add_optimization_profile(profile)

# 5. Build engine
serialized_engine = builder.build_serialized_network(network, config)
if serialized_engine is None:
    raise RuntimeError("Failed to build engine")

with open(ENGINE_PATH, "wb") as f:
    f.write(serialized_engine)

print("TensorRT engine generated:", ENGINE_PATH)






































































# import torch 
# import onnx 
# import tensorrt as trt 
# from torch import nn 
 

# import numpy as np
# import pycuda.driver as cuda
# import pycuda.autoinit

# onnx_model = 'srcnn_48.onnx' 

# class SuperResolutionNet(nn.Module):
#     def __init__(self, upscale_factor):
#         super().__init__()
#         self.upscale_factor = upscale_factor
#         self.img_upsampler = nn.Upsample(
#             scale_factor=self.upscale_factor,
#             mode='bicubic',
#             align_corners=False
#         )
#         self.conv1 = nn.Conv2d(3, 64, kernel_size=9, padding=4)
#         self.conv2 = nn.Conv2d(64, 32, kernel_size=1, padding=0)
#         self.conv3 = nn.Conv2d(32, 3, kernel_size=5, padding=2)
#         self.relu = nn.ReLU()

#     def forward(self, x):
#         x = self.img_upsampler(x)
#         out = self.relu(self.conv1(x))
#         out = self.relu(self.conv2(out))
#         out = self.conv3(out)
#         return out
# device = torch.device('cuda:0') 


# class MyEntropyCalibrator(trt.IInt8EntropyCalibrator2):
#     def __init__(self, batch_size, input_shape, num_batches=10):
#         super().__init__()
#         self.batch_size = batch_size
#         self.input_shape = input_shape  # (B, C, H, W)
#         self.num_batches = num_batches
#         self.current_batch = 0

#         self.device_input = cuda.mem_alloc(
#             int(np.prod(input_shape) * np.float32().nbytes)
#         )

#     def get_batch_size(self):
#         return self.batch_size

#     def get_batch(self, names):
#         if self.current_batch >= self.num_batches:
#             return None

#         # 随机数据（测试用完全够）
#         host_data = np.random.rand(*self.input_shape).astype(np.float32)
#         cuda.memcpy_htod(self.device_input, host_data)

#         self.current_batch += 1
#         return [int(self.device_input)]

#     def read_calibration_cache(self):
#         return None

#     def write_calibration_cache(self, cache):
#         pass
# calibrator = MyEntropyCalibrator(
#     batch_size=48,
#     input_shape=(48, 3, 256, 256),
#     num_batches=10
# )
# # generate ONNX model 
# torch.onnx.export(SuperResolutionNet(upscale_factor=3), torch.randn(48, 3, 256, 256), onnx_model, input_names=['input'], output_names=['output'], dynamo=True , opset_version=18) 
# onnx_model = onnx.load(onnx_model) 
 
# # create builder and network 
# logger = trt.Logger(trt.Logger.ERROR) 
# builder = trt.Builder(logger) 
# EXPLICIT_BATCH = 1 << (int)( 
#     trt.NetworkDefinitionCreationFlag.EXPLICIT_BATCH) 
# network = builder.create_network(EXPLICIT_BATCH) 
 
# # parse onnx 
# parser = trt.OnnxParser(network, logger) 
 
# if not parser.parse(onnx_model.SerializeToString()): 
#     error_msgs = '' 
#     for error in range(parser.num_errors): 
#         error_msgs += f'{parser.get_error(error)}\n' 
#     raise RuntimeError(f'Failed to parse onnx, {error_msgs}') 
 
# config = builder.create_builder_config()
# config.set_memory_pool_limit(
#     trt.MemoryPoolType.WORKSPACE,
#     1 << 30
# )
# if builder.platform_has_fast_fp16:
#     config.set_flag(trt.BuilderFlag.FP16)
# config.set_flag(trt.BuilderFlag.FP16)
# # config.set_flag(trt.BuilderFlag.INT8)
# # config.int8_calibrator = calibrator
# config.profiling_verbosity = trt.ProfilingVerbosity.DETAILED
# profile = builder.create_optimization_profile()
# profile.set_shape(
#     'input',
#     [48, 3, 256, 256],
#     [48, 3, 256, 256],
#     [48, 3, 256, 256]
# )
# config.add_optimization_profile(profile)


# with torch.cuda.device(device):
#     serialized_engine = builder.build_serialized_network(network, config)

# with open('model.engine', 'wb') as f:
#     f.write(serialized_engine)
#     print("generating file done!")
