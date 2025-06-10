import torch
import torchvision
import torchvision.transforms as transforms
import onnxruntime as ort
import numpy as np
import logging
import time
from utils.utils import load_config


def main():
    torch.manual_seed(42)

    config = load_config("config.yaml")
    model_path = config["current_model"]["path"]
    data_path = config['dataset']['paths']['original']['local_path']

    transform = transforms.ToTensor()
    testset = torchvision.datasets.CIFAR10(
        root=data_path, train=False, download=False, transform=transform
    )
    testloader = torch.utils.data.DataLoader(
        testset, batch_size=8, shuffle=False, num_workers=2
    )

    try:
        model = ort.InferenceSession(model_path)
    except Exception as e:
        logging.error(f"Error loading ONNX model: {e}")
        raise e

    input_name = model.get_inputs()[0].name

    correct = 0
    total = 0
    start_time = time.time()

    for images, labels in testloader:
        images_np = images.numpy().astype(np.float32)

        outputs = model.run(None, {input_name: images_np})
        predictions = np.argmax(outputs[0], axis=1)

        correct += np.sum(predictions == labels.numpy())
        total += labels.size(0)

    end_time = time.time()
    test_time = end_time - start_time
    final_accuracy = correct / total * 100

    print(f"Total Inference Time: {test_time:.4f} seconds")
    print(f"Final Test Accuracy: {final_accuracy:.2f}%")


if __name__ == "__main__":
    main()
