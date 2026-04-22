import torch
import torchvision
import torchvision.transforms as transforms
import numpy as np
import cv2

def add_noise(img):
    noise = np.random.normal(0, 0.1, img.shape)
    noisy = img + noise
    return np.clip(noisy, 0, 1)

def build_dataset(n_samples=1000):
    transform = transforms.ToTensor()
    dataset = torchvision.datasets.CIFAR10(root="./data", train=True, download=True, transform=transform)

    X, Y = [], []

    for i in range(n_samples):
        img, _ = dataset[i]
        img_np = img.permute(1,2,0).numpy()

        degraded = add_noise(img_np)

        X.append(torch.tensor(degraded).permute(2,0,1))
        Y.append(img)

    return torch.stack(X), torch.stack(Y)
