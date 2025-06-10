import sys
import os

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
import onnxruntime as ort
import numpy as np
from tqdm import tqdm
from utils.dataloader import data_load
from utils.utils import load_config


def evaluate_model(onnx_model_path, dataloader):
    session = ort.InferenceSession(onnx_model_path)
    input_name = session.get_inputs()[0].name

    correct = 0
    total = 0

    for images, labels in tqdm(dataloader, desc="Evaluating"):
        images = images.numpy()
        outputs = session.run(None, {input_name: images})
        preds = np.argmax(outputs[0], axis=1)
        correct += (preds == labels.numpy()).sum()
        total += labels.size(0)

    return correct / total


def detect_drift(config, threshold=0.05):
    onnx_model_path = config["current_model"]["path"]

    print("Evaluating on original dataset...")
    config['dataset']['version'] = 'original'
    _, original_loader = data_load(config)
    acc_original = evaluate_model(onnx_model_path, original_loader)
    print(f"Original accuracy: {acc_original:.4f}")

    print("Evaluating on augmented dataset...")
    config['dataset']['version'] = 'augmented'
    _, augmented_loader = data_load(config)
    acc_augmented = evaluate_model(onnx_model_path, augmented_loader)
    print(f"Augmented accuracy: {acc_augmented:.4f}")

    drift = acc_original - acc_augmented
    print(f"Accuracy drop: {drift:.4f}")

    if drift > threshold:
        print(f"Model drift detected! Accuracy dropped by more than {threshold:.2f}.")
        return True
    else:
        print("No significant model drift detected.")
        return False


config = load_config("config.yaml")
detect_drift(config, threshold=0.05)
