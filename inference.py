import os
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision.io import read_image
import onnxruntime as ort
import numpy as np
import cv2 as cv

class CustomDataset(Dataset):  # Change to Dataset
    def __init__(self, dir: str = "inference_data", dimensions: tuple[int, int] = (32, 32), transform: bool = True):
        self.dir = dir
        self.data_list = [file for file in os.listdir(self.dir) if os.path.isfile(os.path.join(self.dir, file))]
        self.dims = dimensions
        self.transform_bool = transform
    
    def transform(self, img):
        return cv.resize(img.permute(1, 2, 0).numpy(), self.dims)  # Convert tensor to numpy before resizing

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
    session = ort.InferenceSession("Model_20250209_224001.onnx", 
                                           providers = ['CUDAExecutionProvider', 'CPUExecutionProvider'])
    print("Started ONNX runtime session")
    
    input_name = session.get_inputs()[0].name
    output_name = session.get_outputs()[0].name
    print(f"Session input name: {input_name}")
    
    # iobinding = session.io_binding()
    
    dataset = CustomDataset()
    dataloader = DataLoader(dataset, 1, False)
    print("Initialized data loader")

    for idx, image in enumerate(dataloader):
        img = tensor_to_img(image)
        print(img.shape)
        cv.imshow(f"Image", img)
        image_np = image.numpy().astype(np.float32)
        print(image_np.shape)
        preds = session.run(None, {input_name: image_np})[0]
        print(preds.shape)
        print(preds)
        prediction = np.argmax(preds[0], axis=-1)
        print(f"Prediction: {prediction}")

        cv.waitKey()

if __name__ == "__main__":
    main()