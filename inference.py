import os
import torch
import onnxruntime

def main():
    session = onnxruntime.InferenceSession("Insert ONNX-model path, when available", 
                                           providers = onnxruntime.get_available_providers())
    
    