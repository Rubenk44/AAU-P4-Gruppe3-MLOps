import sys
import os

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
import numpy as np
import matplotlib.pyplot as plt
import torch
import torchvision.models as models
from sklearn.decomposition import PCA
from tqdm import tqdm

from utils.dataloader import data_load
from utils.utils import load_config

config = load_config("config.yaml")

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using device:", device)

train_loader, _ = data_load(config)
resnet = models.resnet50(weights=models.ResNet50_Weights.DEFAULT)
model = torch.nn.Sequential(*list(resnet.children())[:-1]).to(device)
model.eval()

features = []

with torch.no_grad():
    for imgs, _ in tqdm(train_loader, desc="Extracting ResNet50 features"):
        imgs = imgs.to(device)
        out = model(imgs).squeeze()
        features.append(out.cpu().numpy())

features_np = np.vstack(features)

# Flatten (2048-d per image)
if features_np.ndim == 1:
    features_np = features_np.reshape(1, -1)

pca = PCA()
pca.fit(features_np)
explained = np.cumsum(pca.explained_variance_ratio_)
n_components_95 = np.argmax(explained >= 0.95) + 1

plt.figure(figsize=(12, 6))
plt.plot(explained, marker='o', markersize=3)
plt.axhline(y=0.95, color='r', linestyle='--', label='95% Variance')
plt.axvline(
    x=n_components_95, color='g', linestyle='--', label=f'{n_components_95} Components'
)
plt.title('PCA Cumulative Explained Variance')
plt.xlabel('Number of Components')
plt.ylabel('Cumulative Explained Variance')
plt.grid(True)
plt.legend()
plt.tight_layout()
plt.savefig('Lecture6/pca_explained_variance.png')
print("Plot saved as 'pca_explained_variance.png'")
