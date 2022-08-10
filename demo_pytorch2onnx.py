import sys
import onnx
import os
import argparse
import numpy as np
import cv2
import onnxruntime
import torch

from tool.utils import *
from models import Yolov4
from demo_darknet2onnx import detect


def transform_to_onnx(batch_size, n_classes, IN_IMAGE_H, IN_IMAGE_W):
    
    model = Yolov4(n_classes=n_classes, inference=True)

    input_names = ["input"]
    output_names = ['boxes', 'confs']

    x = torch.randn((batch_size, 3, IN_IMAGE_H, IN_IMAGE_W), requires_grad=True)
    onnx_file_name = "yolov4_{}_3_{}_{}_static.onnx".format(batch_size, IN_IMAGE_H, IN_IMAGE_W)
    # Export the model
    torch.onnx.export(model,
                        x,
                        onnx_file_name,
                        export_params=True,
                        opset_version=11,
                        do_constant_folding=True,
                        input_names=input_names, output_names=output_names,
                        dynamic_axes=None)
    


if __name__ == '__main__':
    print("Converting to onnx and running demo ...")
    n_classes = 1000
    batch_size = 1
    IN_IMAGE_H = 416
    IN_IMAGE_W = 416
    transform_to_onnx(batch_size, n_classes, IN_IMAGE_H, IN_IMAGE_W)
