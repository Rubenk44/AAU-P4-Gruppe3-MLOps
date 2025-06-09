import onnx
from onnxruntime.quantization import quantize_dynamic, QuantType


def quantize_onnx_model(
    saved_model_path="models/deepspeed_stage_3/model.onnx",
    output_path="models/deepspeed_stage_3/model_quantized.onnx",
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
