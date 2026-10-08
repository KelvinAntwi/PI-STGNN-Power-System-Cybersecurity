# Physics-Informed Spatio-Temporal GNN for Power-System Cybersecurity

A research implementation of a **Physics-Informed Spatio-Temporal Graph Neural Network (PI-STGNN)** for detecting False Data Injection (FDI) attacks in power-system measurements.

The model combines graph attention, temporal convolution, and AC power-flow constraints to identify abnormal measurements while considering the physical relationships between buses.

## Overview

The current implementation uses the **IEEE 14-bus system** with synthetically generated power-system measurements and simulated FDI attacks.

Each bus is represented by four measurements:

* Active power (P)
* Reactive power (Q)
* Voltage magnitude (|V|)
* Voltage angle (θ)

Measurements are provided as sequences, allowing the model to learn both spatial relationships between buses and temporal patterns in the measurements.

## Model Architecture

The PI-STGNN contains two spatio-temporal blocks. Each block combines:

* Temporal convolution
* Gated temporal convolution
* Multi-head graph attention
* Batch normalization

The learned representation is then used for:

* FDI attack classification
* Power-system state reconstruction

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

The physics component is based on the conductance (G) and susceptance (B) matrices of the IEEE 14-bus network.

The active power at bus *i* is calculated using the AC power-flow relationship:

Pᵢ = Vᵢ Σⱼ₌₁ᴺ Vⱼ [Gᵢⱼ cos(θᵢ − θⱼ) + Bᵢⱼ sin(θᵢ − θⱼ)]

The reactive power is calculated as:

Qᵢ = Vᵢ Σⱼ₌₁ᴺ Vⱼ [Gᵢⱼ sin(θᵢ − θⱼ) − Bᵢⱼ cos(θᵢ − θⱼ)]

The physics loss compares the calculated P and Q values with the corresponding measurements.

The overall training objective is:

ℒ = ℒclass + 0.5ℒrecon + λphysℒphys
where:

λphys = 0.05
This adds a physical consistency constraint to the learning process rather than relying only on the classification error.

## FDI Attack Simulation

The experiment uses synthetic telemetry to evaluate the model.

For attack samples, active-power measurements at selected buses are deliberately perturbed to simulate an FDI attack.

* **Affected measurement:** Active power (P)
* **Selected buses:** 4–6
* **Attack type:** Synthetic false-data injection

The model therefore learns to distinguish between normal and manipulated measurements.

## Experimental Setup

| Parameter                  |       Value |
| -------------------------- | ----------: |
| Test system                | IEEE 14-bus |
| Input features             |           4 |
| Sequence length            |          12 |
| Hidden dimension           |          32 |
| GAT heads                  |           4 |
| Classes                    |           2 |
| Training epochs            |         150 |
| Batch size                 |          32 |
| Optimizer                  |       AdamW |
| Learning rate              |       0.002 |
| Weight decay               |      0.0001 |
| Physics-loss weight        |        0.05 |
| Reconstruction-loss weight |         0.5 |
| Dropout                    |         0.2 |

## Results

The trained model was evaluated on a separate **100-sample test set**.

### Test Performance

| Metric              |     Result |
| ------------------- | ---------: |
| Accuracy            | **81.00%** |
| Classification loss | **0.3626** |
| Physics residual    | **0.3427** |

### FDI Detection Performance

| Class  | Precision | Recall | F1-score |
| ------ | --------: | -----: | -------: |
| Normal |      0.83 |   0.82 |     0.83 |
| Attack |      0.78 |   0.80 |     0.79 |

The model achieved **81% accuracy** on the test set, with an **F1-score of 0.79 for the FDI attack class**.

The training and evaluation curves are saved in:

`results/ac_performance_results.png`

## Repository Structure

```text
PI-STGNN-Power-System-Cybersecurity/
│
├── README.md
├── PI_ST_GNN_Model.ipynb
├── requirements.txt
├── results/
│   └── ac_performance_results.png
├── figures/
└── LICENSE
```

## Running the Notebook

The main implementation is contained in:

`PI_ST_GNN_Model.ipynb`

The notebook can be run in **Google Colab** or a local Python environment.

Install the required packages with:

```bash
pip install -r requirements.txt
```

Then open and run the notebook from start to finish.

## Current Scope

This implementation is an experimental study using the IEEE 14-bus system, synthetic power-system measurements, and simulated FDI attacks.

The current results are intended to evaluate the model architecture and physics-informed learning approach. Further evaluation with larger networks, realistic measurement datasets, and more diverse attack scenarios would be required before practical deployment.

## Research Areas

* Power-system cybersecurity
* False Data Injection attacks
* Graph neural networks
* Spatio-temporal learning
* Physics-informed machine learning
* AC power-system modeling

## Citation

If you use this repository in your research, please cite:

```text
[Add citation when the associated paper is published]
```
