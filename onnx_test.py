import onnx
 
# onnx_model = onnx.load("srcnn.onnx") 
# try: 
#     onnx.checker.check_model(onnx_model) 
# except Exception: 
#     print("Model incorrect") 
# else: 
#     print("Model correct")

import onnxruntime
print(onnxruntime.__version__)
print(onnxruntime.get_device() ) # 如果得到的输出结果是GPU，所以按理说是找到了GPU的

ort_session = onnxruntime.InferenceSession("srcnn.onnx",
providers=['CUDAExecutionProvider'])
print(ort_session.get_providers())