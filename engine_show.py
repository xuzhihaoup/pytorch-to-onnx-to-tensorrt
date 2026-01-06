import tensorrt as trt



logger = trt.Logger(trt.Logger.WARNING)
with open("model.engine", "rb") as f:
    engine = trt.Runtime(logger).deserialize_cuda_engine(f.read())

print("==== Tensor DTypes ====")
for i in range(engine.num_io_tensors):
    name = engine.get_tensor_name(i)
    dtype = engine.get_tensor_dtype(name)
    mode = engine.get_tensor_mode(name)
    print(f"{name:10s} | {mode} | {dtype}")