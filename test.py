import torch
import torchvision
import torchvision.transforms as transforms
import onnxruntime as ort
import numpy as np
import logging

def main():
    torch.manual_seed(42)

    transform = transforms.ToTensor()
    testset = torchvision.datasets.CIFAR10(root='data', train=False, download=True, transform=transform)
    testloader = torch.utils.data.DataLoader(testset, batch_size=8, shuffle=False, num_workers=2)

    model_path = "Model_20250209_224001.onnx"
    try:
        model = ort.InferenceSession(model_path)
    except Exception as e:
        logging.error(f"Error loading ONNX model: {e}")
        raise e

    input_name = model.get_inputs()[0].name 

    total_correct = 0
    total_samples = 0

    for images, labels in testloader:
        images_np = images.numpy().astype(np.float32) 

        outputs = model.run(None, {input_name: images_np})
        predictions = np.argmax(outputs[0], axis=1)

        total_correct += np.sum(predictions == labels.numpy())
        total_samples += labels.size(0)

    final_accuracy = total_correct / total_samples * 100
    print(f"Final Test Accuracy: {final_accuracy:.2f}%")

if __name__ == "__main__":
    main()
