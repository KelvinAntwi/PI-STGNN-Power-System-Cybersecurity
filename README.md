# Physics-Informed Spatio-Temporal GNN for Power-System Cybersecurity

A research implementation of a **Physics-Informed Spatio-Temporal Graph Neural Network (PI-STGNN)** for detecting False Data Injection (FDI) attacks in power-system measurements.

The model combines temporal convolution, multi-head graph attention, and nonlinear AC power-flow constraints to detect manipulated measurements while incorporating the physical relationships between interconnected buses.

## Overview

The current implementation uses the **IEEE 14-bus test system**, synthetically generated power-system measurements, and simulated FDI attacks.

Each bus is represented by four measurement features:

- Active power (P)
- Reactive power (Q)
- Voltage magnitude (|V|)
- Voltage angle (θ)

Measurements are organized into temporal sequences, allowing the model to learn spatial dependencies between buses and temporal patterns in the measurement data.

## Model Architecture

The PI-STGNN comprises two spatio-temporal blocks incorporating:

- Temporal convolution
- Gated temporal convolution
- Multi-head graph attention
- Batch normalization

The learned representation supports two tasks:

1. Binary classification of normal and FDI-manipulated measurements.
2. Power-system state reconstruction for physics-consistency evaluation.

```text
Input: P, Q, |V|, θ
        │
        ▼
Temporal Convolution
        │
        ▼
Gated Temporal Convolution
        │
        ▼
Multi-Head Graph Attention
        │
        ▼
Spatio-Temporal Block × 2
        │
        ├──────────────► Attack Classification
        │
        └──────────────► State Reconstruction
                                │
                                ▼
                       Physics Consistency
```

## Physics-Informed Component

The physics-informed component uses the conductance (G) and susceptance (B) matrices derived from the IEEE 14-bus network admittance matrix.

The active-power injection at bus *i* is calculated using the nonlinear AC power-flow relationship:

\[
P_i = V_i \sum_{j=1}^{N} V_j
\left[
G_{ij}\cos(\theta_i-\theta_j)
+
B_{ij}\sin(\theta_i-\theta_j)
\right]
\]

The corresponding reactive-power relationship is:

\[
Q_i = V_i \sum_{j=1}^{N} V_j
\left[
G_{ij}\sin(\theta_i-\theta_j)
-
B_{ij}\cos(\theta_i-\theta_j)
\right]
\]

The physics-consistency loss compares the calculated active and reactive power values with their corresponding measurements.

The overall training objective is:

\[
\mathcal{L}
=
\mathcal{L}_{class}
+
0.5\mathcal{L}_{recon}
+
\lambda_{phys}\mathcal{L}_{phys}
\]

where:

- \(\mathcal{L}_{class}\): Classification loss
- \(\mathcal{L}_{recon}\): Reconstruction loss
- \(\mathcal{L}_{phys}\): Physics-consistency loss
- \(\lambda_{phys}=0.05\): Physics-loss weighting coefficient

This formulation introduces a physical consistency constraint alongside the classification and reconstruction objectives.

## FDI Attack Simulation

The experiment uses synthetic telemetry to simulate normal operating measurements and false-data-injection attacks.

For attack samples, active-power measurements at selected buses are deliberately perturbed.

| Parameter | Description |
|---|---|
| Attack type | Synthetic false-data injection |
| Affected measurement | Active power (P) |
| Selected buses | 4–6 |
| Classification task | Normal vs. attack |

The current experiment focuses on detecting these simulated measurement manipulations rather than evaluating the full range of possible power-system cyberattacks.

## Experimental Setup

| Parameter | Value |
|---|---:|
| Test system | IEEE 14-bus |
| Input features | 4 |
| Sequence length | 12 |
| Hidden dimension | 32 |
| GAT heads | 4 |
| Classes | 2 |
| Training epochs | 150 |
| Batch size | 32 |
| Optimizer | AdamW |
| Learning rate | 0.002 |
| Weight decay | 0.0001 |
| Physics-loss weight | 0.05 |
| Reconstruction-loss weight | 0.5 |
| Dropout | 0.2 |
| Held-out test samples | 100 |

## Experimental Results

The PI-STGNN was trained for 150 epochs and subsequently evaluated on a separate synthetic test set containing 100 samples: 55 normal and 45 attack samples.

The results below correspond to the latest reported experimental run.

### 1. Training Performance

At epoch 150, the model recorded:

| Training Metric | Result |
|---|---:|
| Training accuracy | 81.25% |
| Attack F1-score | 81.25% |
| Total loss | 0.7294 |
| Classification loss | 0.4004 |
| Physics residual | 0.3666 |

The training-performance figure illustrates the evolution of classification loss, physics residual, detection accuracy, and attack F1-score across the training epochs.

**Figure 1. PI-STGNN Training Performance on the IEEE 14-Bus System**

<img width="989" height="592" alt="PI-STGNN training performance showing classification loss, physics residual, accuracy and attack F1-score across epochs" src="https://github.com/user-attachments/assets/1aded6fb-8143-47bc-ae2a-63f359bf7a90" />

*Note: Training metrics are calculated on generated training batches. The final-epoch values should not be interpreted as aggregate performance across the entire training dataset.*

### 2. Held-Out Test Performance

The trained model was evaluated on a separate 100-sample synthetic test set.

| Evaluation Metric | Result |
|---|---:|
| Test accuracy | **83.00%** |
| Classification loss | 0.3788 |
| Physics residual | 0.3404 |

**Class-wise Classification Performance**

| Class | Precision | Recall | F1-score | Support |
|---|---:|---:|---:|---:|
| Normal (0) | 0.83 | 0.87 | 0.85 | 55 |
| Attack (1) | 0.83 | 0.78 | 0.80 | 45 |

The model correctly classified 83 of the 100 test samples, achieving an attack-class precision of approximately 83%, recall of 78%, and F1-score of 80%.

These results describe performance on the simulated test distribution and should not be interpreted as validation on operational power-grid measurements.

### 3. Test-Set Confusion Matrix

The confusion matrix summarizes the model's predictions on the held-out test set.

| Actual / Predicted | Normal (0) | Attack (1) |
|---|---:|---:|
| Normal (0) | 48 | 7 |
| Attack (1) | 10 | 35 |

The evaluation recorded:

- 48 correctly classified normal samples.
- 35 correctly detected attack samples.
- 7 normal samples incorrectly classified as attacks.
- 10 attack samples incorrectly classified as normal.

**Figure 2. PI-STGNN Test-Set Confusion Matrix**

<img width="733" height="590" alt="image" src="https://github.com/user-attachments/assets/acf53c39-c9a4-4217-9786-7b94ffae3ce2" />


*The confusion matrix visualizes the classification results from the held-out test set, independently of the training-performance curves.*

## Running the Notebook

The PI-STGNN implementation is available in the repository notebook:

[**PI_ST_GNN_Model.ipynb**](PI_ST_GNN_Model.ipynb)

Open the notebook to inspect the model architecture, synthetic data generation, training procedure, physics-informed loss implementation, and final evaluation.

The notebook can also be opened in Google Colab for execution.

### Reproducibility

The experiment involves random model initialization and synthetic data generation. Consequently, results may vary between independent executions unless random seeds and the computational environment are controlled.

The reported metrics correspond to the documented experimental run. For more comprehensive performance assessment, future experiments should include repeated runs with controlled random seeds and report aggregate statistics.

## Current Scope and Limitations

This implementation is an experimental investigation of physics-informed spatio-temporal graph learning for power-system cybersecurity.

Its current scope includes:

- Evaluation on the IEEE 14-bus test system.
- Synthetically generated measurement sequences.
- Simulated FDI attacks affecting selected active-power measurements.
- Binary classification of normal and manipulated measurements.
- Physics-informed learning using nonlinear AC power-flow relationships.

Further investigation is required to assess generalization across larger power networks, realistic operating conditions, measurement noise, different attack strategies, and real-world power-system datasets.

The reported results demonstrate performance under the defined experimental conditions and do not establish deployment readiness.

## Citation

Antwi, N. K. (2026). *Physics-Informed Spatio-Temporal Graph Neural Network for Power-System Cybersecurity*. GitHub repository.

https://github.com/KelvinAntwi/PI-STGNN-Power-System-Cybersecurity

This repository contains a research implementation developed as part of ongoing work on physics-informed graph neural networks for power-system cybersecurity.
