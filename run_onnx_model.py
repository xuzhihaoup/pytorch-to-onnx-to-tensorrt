import onnxruntime 
import cv2 
import numpy as np
import time




input_img = cv2.imread('face.png').astype(np.float32) 
print(f"Input image shape: {input_img.shape}")
# HWC to NCHW 
input_img = np.transpose(input_img, [2, 0, 1]) 
input_img = np.expand_dims(input_img, 0)  
input_batch = np.repeat(input_img, repeats=1, axis=0)
print("Batch shape:", input_batch.shape) 
device = 'CPUExecutionProvider'
ort_session = onnxruntime.InferenceSession(
    "srcnn_1.onnx", 
    providers=[device]  # 指定使用 GPU  ['TensorrtExecutionProvider', 'CUDAExecutionProvider', 'CPUExecutionProvider']
)
print('Using device: {}'.format(device))
ort_inputs = {'input': input_batch} 
st = time.time()
ort_output = ort_session.run(['output'], ort_inputs)[0] 
et = time.time()
print(f"Inference time: {et - st:.4f} seconds")
# ort_output = np.squeeze(ort_output, 0) 
ort_output = np.clip(ort_output[0], 0, 255) 
ort_output = np.transpose(ort_output, [1, 2, 0]).astype(np.uint8) 
cv2.imwrite("face_onnx.png", ort_output)