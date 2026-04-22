# Hybrid Quantum-Classical Image Denoising (CNN and VQC)

This project explores a hybrid deep learning approach that combines Convolutional Neural Networks (CNNs) with Variational Quantum Circuits (VQC) for image denoising. The goal is to investigate whether quantum feature transformations can enhance classical image restoration performance.


## Objective

* Restore images degraded by noise using a hybrid model
* Integrate a quantum layer within a CNN architecture
* Compare performance with classical approaches using quantitative metrics



## Methodology

### Pipeline

Degraded Image → CNN Encoder → Feature Vector → Quantum Layer → CNN Decoder → Restored Image

### Key Components

* **CNN Encoder**: Extracts spatial features from noisy images
* **Quantum Layer (VQC)**:

  * Encodes features into qubits
  * Applies parameterized quantum gates
  * Returns transformed features
* **CNN Decoder**: Reconstructs the denoised image



## How to Run

### 1. Clone the repository

```bash
git clone https://github.com/Shreya-230206/hybrid_quantum_cnn.git
cd hybrid-quantum-cnn
```

### 2. Install dependencies

```bash
pip install -r requirements.txt
```

### 3. Train the model

```bash
python train.py
```

### 4. Test the model

```bash
python test.py
```



## Evaluation Metrics

* **PSNR (Peak Signal-to-Noise Ratio)**
* **SSIM (Structural Similarity Index)**

These metrics are used to compare:

* Degraded Image
* CNN Output
* Hybrid CNN + Quantum Output



## Key Insights

* Hybrid quantum-classical models can be integrated into deep learning pipelines
* Quantum layers act as feature transformation modules
* Performance gains may vary, but the approach demonstrates a novel research direction




## Author

Shreya Kumari



## Note

This project is an experimental study combining classical deep learning with quantum computing concepts for research and learning purposes.
