# -*- coding: utf-8 -*-
"""把 erquren resnet34 车型识别模型导出为 ONNX (一次性)"""
import sys
sys.path.insert(0, ".")
import torch
from model import resnet34

model = resnet34(num_classes=1778, include_top=True)
model.load_state_dict(torch.load("resnet_car.pth", map_location="cpu"))
model.eval()

x = torch.randn(1, 3, 224, 224)
torch.onnx.export(
    model, x, "vehicle_brand.onnx",
    input_names=["input"], output_names=["logits"],
    opset_version=17,
    dynamic_axes={"input": {0: "batch"}, "logits": {0: "batch"}},
)
print("exported vehicle_brand.onnx")
