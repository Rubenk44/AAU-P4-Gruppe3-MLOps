import sys
import os

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
import time
import onnxruntime as ort
import numpy as np
from utils.dataloader import data_load
from utils.utils import load_config


def to_numpy(tensor):
    return (
        tensor.detach().cpu().numpy() if tensor.requires_grad else tensor.cpu().numpy()
    )


def benchmark_onnx_model():
    config = load_config("config.yaml")
    val_loader = data_load(config)[1]
    model_path = config["current_model"]["path"]

    session = ort.InferenceSession(model_path, providers=["CPUExecutionProvider"])

    input_name = session.get_inputs()[0].name
    output_name = session.get_outputs()[0].name

    total = correct = 0
    start = time.time()

    for inputs, labels in val_loader:
        inputs = to_numpy(inputs)
        outputs = session.run([output_name], {input_name: inputs})[0]

        preds = np.argmax(outputs, axis=1)
        correct += (preds == labels.numpy()).sum()
        total += labels.size(0)

    elapsed = time.time() - start
    accuracy = 100.0 * correct / total
    print(f"Inference time: {elapsed:.2f}s, Accuracy: {accuracy:.2f}%")


if __name__ == "__main__":
    benchmark_onnx_model()
