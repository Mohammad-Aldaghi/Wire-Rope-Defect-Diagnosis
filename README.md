# Physics-Informed Deep Learning for MFL Wire Rope Inspection 🚠

> **An Uncertainty-Aware Sensor Fusion Framework for Single and Multi-Break Defect Diagnosis**

This repository contains the official implementation of our framework for the automated diagnosis and classification of steel wire rope defects using Magnetic Flux Leakage (MFL) signals.

## 🌟 Overview & Achievements
In industrial inspections, MFL signals become highly distorted when encountering overlapping multi-wire breaks. Instead of relying on noisy, unlabelled field data, this project introduces a **Physics-Informed Forward Model** based on global **OITAF** standards to simulate thousands of complex fault signatures with high fidelity.

By leveraging two isolated **1D-CNNs** and a **Variance-based Fusion Engine**, this framework effectively extracts the hidden physical parameters of wire rope degradation directly from raw 1D signals.

### ✨ Key Features:
* **Uncertainty Estimation:** Utilizes *Monte Carlo Dropout* and *Gaussian Negative Log-Likelihood (GNLL)* loss to evaluate the model's epistemic uncertainty during predictions.
* **Multi-Break Support:** Capable of isolating and predicting 1 to 6 consecutive broken wires and measuring the physical distance between them.
* **Robust Accuracy:** Achieves **83.5%** Top-1 accuracy and **98.1%** Top-2 accuracy in distinguishing Inner and Outer fault topologies.
* **Smart Sensor Fusion:** Automatically penalizes and filters out physically illogical predictions without requiring manual user thresholds.

---

## 📊 Architecture & Results

<!-- 
Image Upload Guide:
1. Create a folder named `docs` in your GitHub repository.
2. Upload your flowchart as `flowchart.png` and your 4x4 matrix as `confusion_matrix.png` inside the `docs` folder.
3. The images will automatically render below once uploaded!
-->

![Flowchart](docs/flowchart.png)
*Project Flowchart: From physics-informed dataset generation to uncertainty-aware fusion.*

![Confusion Matrix](docs/confusion_matrix.png)
*4x4 Macro-Class Confusion Matrix showcasing the model's high precision in distinguishing Inner and Outer topologies across both single and multi-break domains.*

---

## 🛠️ Getting Started

Follow these instructions to set up and run the project on your local machine.

### 1. Prerequisites
Ensure Python is installed, then install the required dependencies:
```bash
pip install torch numpy pandas scikit-learn matplotlib seaborn
```

### 2. Dataset Generation
Due to storage constraints, the large synthetic dataset is not included in this repository. Generate the physical Single and Multi-break data by running:
```bash
python src/dataset_generator.py
```
*(Generated `.npz` files will be saved in the designated data directory).*

### 3. Model Training
To train the probabilistic 1D-CNN models from scratch, run:
```bash
python src/train_model.py
```
*(Trained weights will be saved as `.pth` files).*

### 4. Evaluation & Fusion Engine
To evaluate the Fusion Engine, plot the 4x4 confusion matrix, and view the Top-3 probabilistic predictions, execute:
```bash
python src/evaluator.py
```

---
**Note:** The models in this repository are strictly optimized for standard hauling ropes (Inner and Outer breaks). Track-rope specific faults (e.g., King wire breaks) have been filtered out of this version.
