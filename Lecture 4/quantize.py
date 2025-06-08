import os
import sys
import onnx
from onnxruntime.quantization import quantize_dynamic, QuantType

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))


def quantize_onnx_model(
    saved_model_path="../mlops/Model_20250606_230312.onnx",
    output_path="Lecture 4/quantized_model.onnx",
):
    print(f"Loading ONNX model from: {saved_model_path}")
    model = onnx.load(saved_model_path)
    onnx.checker.check_model(model)

    print("Applying dynamic quantization...")
    quantize_dynamic(
        model_input=saved_model_path,
        model_output=output_path,
        weight_type=QuantType.QUInt8,
    )

    print(f"Quantized ONNX model saved to: {output_path}")


if __name__ == "__main__":
    quantize_onnx_model()
