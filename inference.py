import os
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision.io import read_image
import onnxruntime as ort
import numpy as np
import cv2 as cv


class CustomDataset(Dataset):
    def __init__(
        self,
        dir: str = "inference_data",
        transform: bool = True,
        dimensions: tuple[int, int] | list[int, int] = (32, 32),
    ):
        """
        Instantiates a custom dataset with the default directory being 'inference_data'.

        Attributes
        -------------
        dir: str
            the directory in which the custom dataset is stored
        transform: bool
            logic for enabling or disabling the images to a certain size, see dimensions
        dimensions: tuple[int, int] | list[int, int]
            the dimensions for the optional resize, given by the transform boolean
        -------------
        """
        self.dir = dir
        self.data_list = [
            file
            for file in os.listdir(self.dir)
            if os.path.isfile(os.path.join(self.dir, file))
        ]
        self.dims = dimensions
        self.transform_bool = transform

    def transform(self, img):
        return cv.resize(
            img.permute(1, 2, 0).numpy(), self.dims
        )  # Convert tensor to numpy before resizing

    def __len__(self):
        return len(self.data_list)

    def __getitem__(self, idx):
        img_path = os.path.join(self.dir, self.data_list[idx])
        img = read_image(img_path).float()  # Read image as float tensor
        if self.transform_bool:
            img = self.transform(img)
            img = torch.from_numpy(img).permute(2, 0, 1)  # Convert back to tensor
        return img


def tensor_to_img(tensor):
    array = tensor.squeeze(0)
    array = array.detach().cpu().numpy()
    array = array.transpose(1, 2, 0)
    return array.astype(np.uint8)


def main():
    # Starts the inference session
    session = ort.InferenceSession(
        "Model_20250209_224001.onnx",
        providers=['CUDAExecutionProvider', 'CPUExecutionProvider'],
    )
    print("Started ONNX runtime session")
    input_name = session.get_inputs()[0].name
    print(f"Session input name: {input_name}")

    # Instantiates the custom datasat and loader
    dataset = CustomDataset()
    dataloader = DataLoader(dataset, 1, False)
    print("Initialized data loader")

    # Performs inference for each picture in the custom dataset
    for idx, image in enumerate(dataloader):
        # First transforming the image to a numpy array for use in openCV
        img = tensor_to_img(image)
        print(img.shape)
        cv.imshow("Image", img)
        # Also transforms the tensor to numpy for use in the ONNX session
        image_np = image.numpy().astype(np.float32)
        preds = session.run(None, {input_name: image_np})[0]
        print(preds.shape)
        print(preds)
        prediction = np.argmax(preds[0], axis=-1)
        print(f"Prediction: {prediction}")

        cv.waitKey()


if __name__ == "__main__":
    main()
