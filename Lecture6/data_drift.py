import sys
import os

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
import numpy as np
import torch
import torchvision.models as models
from torchvision import transforms
from sklearn.decomposition import PCA
from tqdm import tqdm
from scipy.spatial.distance import jensenshannon
from utils.dataloader import data_load
from utils.utils import load_config


def extract_features(model, loader, device):
    features = []
    model.eval()
    with torch.no_grad():
        for images, _ in tqdm(loader, desc="Extracting features", ncols=100):
            images = images.to(device)
            outputs = model(images).squeeze(-1).squeeze(-1)
            features.append(outputs.cpu())
    return torch.cat(features, dim=0).numpy()


def compute_multivariate_jsd(ref, test, bins=10):
    jsd_total = 0
    component_jsds = []

    for i in range(ref.shape[1]):
        ref_counts, bin_edges = np.histogram(ref[:, i], bins=bins)
        test_counts, _ = np.histogram(test[:, i], bins=bin_edges)

        ref_sum = np.sum(ref_counts)
        test_sum = np.sum(test_counts)

        if ref_sum == 0 or test_sum == 0:
            continue

        ref_dist = ref_counts / ref_sum
        test_dist = test_counts / test_sum

        ref_dist = np.clip(ref_dist, 1e-8, 1)
        test_dist = np.clip(test_dist, 1e-8, 1)

        jsd = jensenshannon(ref_dist, test_dist, base=2) ** 2
        jsd_total += jsd
        component_jsds.append(jsd)

    if not component_jsds:
        return 0.0, []

    avg_jsd = jsd_total / len(component_jsds)
    return avg_jsd


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    config = load_config("config.yaml")

    # Force transform override
    transform = transforms.Compose(
        [
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ]
    )
    config['custom_transform'] = transform

    # Load reference/original CIFAR
    config['dataset']['version'] = 'original'
    original_loader, _ = data_load(config)

    # Load augmented CIFAR
    config['dataset']['version'] = 'augmented'
    augmented_loader, _ = data_load(config)

    resnet = models.resnet50(weights=models.ResNet50_Weights.DEFAULT)
    model = torch.nn.Sequential(*list(resnet.children())[:-1]).to(device)

    ref_features = extract_features(model, original_loader, device)
    test_features = extract_features(model, augmented_loader, device)

    pca = PCA(n_components=1142)
    ref_pca = pca.fit_transform(ref_features)
    test_pca = pca.transform(test_features)

    avg_jsd = compute_multivariate_jsd(ref_pca, test_pca)

    print("\nJSD Computation Complete")
    print(f"Average JSD: {avg_jsd:.4f}")


if __name__ == "__main__":
    main()
