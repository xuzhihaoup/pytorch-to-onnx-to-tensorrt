import os
import cv2
import numpy as np
import requests
import torch
from torch import nn
import time
from torch.export import Dim
class SuperResolutionNet(nn.Module):
    def __init__(self, upscale_factor):
        super().__init__()
        self.upscale_factor = upscale_factor
        self.img_upsampler = nn.Upsample(
            scale_factor=self.upscale_factor,
            mode='bicubic',
            align_corners=False
        )
        self.conv1 = nn.Conv2d(3, 64, kernel_size=9, padding=4)
        self.conv2 = nn.Conv2d(64, 32, kernel_size=1, padding=0)
        self.conv3 = nn.Conv2d(32, 3, kernel_size=5, padding=2)
        self.relu = nn.ReLU()

    def forward(self, x):
        x = self.img_upsampler(x)
        out = self.relu(self.conv1(x))
        out = self.relu(self.conv2(out))
        out = self.conv3(out)
        return out

# 下载模型权重和测试图片
urls = [
    'https://download.openmmlab.com/mmediting/restorers/srcnn/srcnn_x4k915_1x16_1000k_div2k_20200608-4186f232.pth',
    'https://raw.githubusercontent.com/open-mmlab/mmediting/master/tests/data/face/000001.png'
]
names = ['srcnn.pth', 'face.png']
for url, name in zip(urls, names):
    if not os.path.exists(name):
        open(name, 'wb').write(requests.get(url).content)

def init_torch_model(device='cuda'):
    torch_model = SuperResolutionNet(upscale_factor=3)
    state_dict = torch.load('srcnn.pth', map_location=device)['state_dict']
    # 调整 key
    for old_key in list(state_dict.keys()):
        new_key = '.'.join(old_key.split('.')[1:])
        state_dict[new_key] = state_dict.pop(old_key)
    torch_model.load_state_dict(state_dict)
    return torch_model.to(device).eval()

device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Using device: {device}")
model = init_torch_model(device)

# 读取图片
input_img = cv2.imread('face.png').astype(np.float32)
print(f"Input image shape: {input_img.shape}")

# HWC -> NCHW，并转为 tensor 放到 GPU
input_tensor = torch.from_numpy(np.transpose(input_img, [2, 0, 1])).unsqueeze(0).to(device)

# 推理
torch.cuda.synchronize() if device == "cuda" else None
start_time = time.time()
with torch.no_grad():
    torch_output = model(input_tensor)
torch.cuda.synchronize() if device == "cuda" else None
end_time = time.time()
print(f"Inference time: {end_time - start_time:.4f} seconds")

# 转回 CPU 并转为 numpy
torch_output = torch_output.cpu().squeeze(0).clamp(0, 255).numpy()
torch_output = np.transpose(torch_output, [1, 2, 0]).astype(np.uint8)

# 保存结果
print(f"Output image shape: {torch_output.shape}")
cv2.imwrite("face_torch.png", torch_output)

# 导出 ONNX
x = torch.randn(1, 3, 256, 256).to(device)

with torch.no_grad():
    torch.onnx.export(
        model,
        x,
        "srcnn_1.onnx",
        opset_version=18,
        input_names=["input"],     # ONNX 名字
        output_names=["output"],   # ONNX 名字
        dynamo=True,
        # dynamic_shapes={
        #     "x": {                 #forward 参数名
        #         0: Dim("batch", min=1)
        #     }
        # }
    )
print("ONNX export done!")
