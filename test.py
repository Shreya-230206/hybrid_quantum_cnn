import torch
import numpy as np
import matplotlib.pyplot as plt
from model import HybridModel
from data_pipeline import build_dataset
from skimage.metrics import peak_signal_noise_ratio as psnr
from skimage.metrics import structural_similarity as ssim

model = HybridModel()
model.load_state_dict(torch.load("hybrid_model.pth"))
model.eval()

X, Y = build_dataset(5)

for i in range(5):
    inp = X[i].unsqueeze(0)
    output = model(inp)[0].detach()

    degraded = X[i].permute(1,2,0).numpy()
    original = Y[i].permute(1,2,0).numpy()
    restored = output.permute(1,2,0).numpy()

    restored = np.clip(restored, 0, 1)

    print("PSNR:", psnr(original, restored))
    print("SSIM:", ssim(original, restored, channel_axis=2, data_range=1.0))

    plt.figure(figsize=(10,3))

    plt.subplot(1,3,1)
    plt.imshow(original)
    plt.title("Original")
    plt.axis("off")

    plt.subplot(1,3,2)
    plt.imshow(degraded)
    plt.title("Degraded")
    plt.axis("off")

    plt.subplot(1,3,3)
    plt.imshow(restored)
    plt.title("Hybrid Output")
    plt.axis("off")

    plt.show()
