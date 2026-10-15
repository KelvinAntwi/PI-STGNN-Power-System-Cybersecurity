# Physics-Informed Spatio-Temporal GNN for Power-System Cybersecurity

A research implementation of a **Physics-Informed Spatio-Temporal Graph Neural Network (PI-STGNN)** for detecting false data injection (FDI) attacks in power-system measurements.

The model combines temporal convolution, multi-head graph attention, and nonlinear AC power-flow constraints to detect manipulated measurements while incorporating physical relationships between interconnected buses.

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

Pᵢ = Vᵢ Σⱼ₌₁ᴺ Vⱼ [Gᵢⱼ cos(θᵢ − θⱼ) + Bᵢⱼ sin(θᵢ − θⱼ)]

The corresponding reactive-power relationship is:

Qᵢ = Vᵢ Σⱼ₌₁ᴺ Vⱼ [Gᵢⱼ sin(θᵢ − θⱼ) − Bᵢⱼ cos(θᵢ − θⱼ)]

The physics-consistency loss compares the calculated active and reactive power values with their corresponding measurements.

The overall training objective is:

ℒ = ℒclass + 0.5ℒrecon + λphysℒphys

where:

- ℒclass: Classification loss
- ℒrecon: Reconstruction loss
- ℒphys: Physics-consistency loss
- λphys = 0.05: Physics-loss weighting coefficient

This formulation introduces a physical consistency constraint alongside the classification and reconstruction objectives.

## FDI Attack Simulation

The experiment uses synthetic telemetry to simulate normal operating measurements and false data injection attacks.

For attack samples, active-power measurements at selected buses are deliberately perturbed.

| Parameter | Description |
|---|---|
| Attack type | Synthetic false data injection |
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

The PI-STGNN was trained for 150 epochs and evaluated on a held-out synthetic test set containing 100 samples: 55 normal samples and 45 FDI attack samples.

The results below correspond to the latest reported experimental run.

### 1. Training Performance

At epoch 150, the model recorded the following training-batch metrics:

| Training Metric | Result |
|---|---:|
| Training accuracy | 78.12% |
| Attack F1-score | 77.42% |
| Total loss | 0.7511 |
| Classification loss | 0.4263 |
| Physics residual | 0.3123 |

<img width="989" height="592" alt="image" src="https://github.com/user-attachments/assets/ae80e0a2-bd49-4419-a28b-dd50b081f178" />

### 2. Held-Out Test Performance

The trained model was evaluated on a separate 100-sample synthetic test set.

| Evaluation Metric | Result |
|---|---:|
| Test accuracy | **79.00%** |
| Classification loss | 0.4386 |
| Physics residual | 0.4205 |

**Class-wise Classification Performance**

| Class | Precision | Recall | F1-score | Support |
|---|---:|---:|---:|---:|
| Normal (0) | 0.93 | 0.67 | 0.78 | 55 |
| Attack (1) | 0.70 | 0.93 | 0.80 | 45 |
| Overall accuracy | — | — | 0.79 | 100 |

The model achieved 79% overall test accuracy and an 80% F1-score for the attack class. Attack recall was 93%, indicating that most attack samples in this test set were detected. However, normal-class recall was 67%, meaning that some normal samples were incorrectly flagged as attacks.

These results illustrate the model's detection performance under the defined synthetic test conditions. They do not establish performance on operational power-grid measurements.

### 3. Test-Set Confusion Matrix

The confusion matrix summarizes the model's predictions on the held-out test set.

| Actual / Predicted | Normal (0) | Attack (1) |
|---|---:|---:|
| Normal (0) | 37 | 18 |
| Attack (1) | 3 | 42 |

The evaluation recorded:

- 37 correctly classified normal samples.
- 42 correctly detected attack samples.
- 18 normal samples incorrectly classified as attacks.
- 3 attack samples incorrectly classified as normal.

<img width="733" height="590" alt="image" src="https://github.com/user-attachments/assets/eaaf7f5f-937e-4119-a943-42473d130734" />

## Running the Notebook

The PI-STGNN implementation is available in the repository notebook:

[**PI_ST_GNN_Model.ipynb**](PI_ST_GNN_Model.ipynb)

Open the notebook to inspect the model architecture, synthetic data generation, training procedure, physics-informed loss implementation, and final evaluation.

## Reproducibility

The experiment involves random model initialization and synthetic data generation. Consequently, results may vary between independent executions unless random seeds and the computational environment are controlled.

The reported metrics correspond to the experimental run documented above. More comprehensive performance assessment would require repeated runs under controlled conditions, with aggregate statistics reported across runs.

## Current Scope and Limitations

This implementation is an experimental investigation of physics-informed spatio-temporal graph learning for power-system cybersecurity.

Its current scope includes:

- Evaluation on the IEEE 14-bus test system.
- Synthetically generated measurement sequences.
- Simulated FDI attacks affecting selected active-power measurements.
- Binary classification of normal and manipulated measurements.

Further investigation is required to assess generalization across larger power networks, realistic operating conditions, measurement noise, different attack strategies, and real-world power-system datasets.

The reported results demonstrate performance under the defined experimental conditions and do not establish deployment readiness.

## Researcher Contributions

I independently conducted the research, developed and implemented the Physics-Informed Spatio-Temporal Graph Neural Network (PI-STGNN), incorporated power-system physics constraints, designed and ran experiments on the IEEE 14-bus test system, and evaluated the model’s performance in detecting false data injection attacks. Mr. Godfred Kofi Junior Tano served as my supervisor, reviewing the work and providing academic oversight.

## Citation

Antwi, N. K. (2026). *Physics-Informed Spatio-Temporal Graph Neural Network for Power-System Cybersecurity*. GitHub repository.

https://github.com/KelvinAntwi/PI-STGNN-Power-System-Cybersecurity

This repository contains a research implementation developed as part of ongoing work on physics-informed graph neural networks for power-system cybersecurity.
