import time
import onnxruntime as ort
import numpy as np
from utils.dataloader import data_load
from utils.utils import load_config


def to_numpy(tensor):
    return (
        tensor.detach().cpu().numpy() if tensor.requires_grad else tensor.cpu().numpy()
    )


def benchmark_onnx_model(
    model_path="models/deepspeed_stage_1/model_quantized.onnx",
    config_path="config.yaml",
    batch_size=32,
):
    config = load_config(config_path)
    val_loader = data_load(config)[1]

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
