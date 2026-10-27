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

![Flowchart](docs/flowchart.png)
*Project Flowchart: From physics-informed dataset generation to uncertainty-aware fusion.*

![Confusion Matrix](docs/confusion_matrix.png)
*4x4 Macro-Class Confusion Matrix showcasing the model's high precision in distinguishing Inner and Outer topologies across both single and multi-break domains.*

---

## 🛠️ Getting Started

Follow these instructions to set up and run the project on your local machine.

### 1. Prerequisites
Ensure Python and Jupyter Notebook are installed, then install the required dependencies:
```bash
pip install torch numpy pandas scikit-learn matplotlib seaborn jupyter
```

### 2. Dataset Generation
Due to storage constraints, the large synthetic dataset is not included in this repository. If you want to generate the physical Single and Multi-break data from scratch, open and run the following Jupyter Notebook:
* `generate_dataset.ipynb`

*(Generated `.npz` files will be automatically saved in the designated data directory).*

### 3. Model Training & Evaluation
If you want to use our pre-prepared models, train the probabilistic 1D-CNN networks from scratch, or evaluate the Fusion Engine's performance, open and run:
* `train_model.ipynb`

*(This notebook handles the entire pipeline: training, weight saving, running the Fusion Engine, and plotting the evaluation metrics).*

---

## 📄 Citation

If you find this code useful in your research or applications, please consider citing our paper:

> **Aldaghi, M., Mohammadi, S., & Mollazadeh Ghomi, M. (2026).** *A Physics-Informed Deep Learning Fusion Engine for Magnetic Flux Leakage Signal Analysis in Wire Ropes.* ISAV 2026 (International Conference on Acoustics and Vibration).

---
**⚠️ Note on Fault Classes (King Wire):** 
The results and confusion matrix presented in this repository and our corresponding paper are specifically filtered for standard hauling ropes (Inner and Outer breaks). However, the core 1D-CNN architectures and Fusion Engine were fundamentally designed and trained to include **King wire (core)** breaks as well. You can easily enable and evaluate the King wire predictions by modifying the class filters in the evaluation scripts/notebooks.
