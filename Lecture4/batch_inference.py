import time
import numpy as np
import onnxruntime as ort
from carbontracker.tracker import CarbonTracker


def batch_infer(
    model_path="models/deepspeed_stage_1/model.onnx", batch_size=32, num_batches=5000
):

    tracker = CarbonTracker(epochs=1)
    tracker.epoch_start()

    session = ort.InferenceSession(model_path, providers=["CPUExecutionProvider"])

    input_name = session.get_inputs()[0].name
    input_shape = session.get_inputs()[0].shape
    input_dtype = np.float32

    input_shape = [
        batch_size if isinstance(dim, str) or dim is None else dim
        for dim in input_shape
    ]

    dummy_input = np.random.randn(*input_shape).astype(input_dtype)

    print("Running ONNX batch inference...")
    start = time.time()
    for _ in range(num_batches):
        _ = session.run(None, {input_name: dummy_input})
    end = time.time()

    avg_time = (end - start) / num_batches
    print(f"Avg batch inference time (batch_size={batch_size}): {avg_time:.4f}s")

    tracker.epoch_end()


if __name__ == "__main__":
    batch_infer()
