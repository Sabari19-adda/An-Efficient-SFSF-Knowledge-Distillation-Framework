# A-Selective-Feature-Layer-and-Soft-Label-Fused-Knowledge-Distillation-System-
Designed and implemented a Selective Feature-Layer and Soft-Label Fused Knowledge Distillation (SFSF-KD) framework (System 100) for lightweight medical image classification, specifically multi-stage Alzheimer's disease detection from brain MRI scans. 


# Selective Feature-Layer and Soft-Label Fused Knowledge Distillation (SFSF-KD)

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![TensorFlow 2.x](https://img.shields.io/badge/TensorFlow-2.x-orange.svg)](https://tensorflow.org/)

An end-to-end framework for transferring hierarchical knowledge from high-capacity teacher models (InceptionV3 / ResNet101) to an ultra-lightweight student neural network using non-consecutive feature layer alignment and soft-label supervision for multi-stage Alzheimer's Disease MRI classification.

---

## 📌 Abstract

Deploying deep learning models in resource-constrained medical environments is hindered by massive memory footprints and heavy computational demands. **SFSF-KD** resolves this by fusing soft-label probability distributions (via temperature-scaled KL divergence) with selective intermediate and deep semantic feature transfer from non-consecutive teacher layers. The framework achieves high-ratio model compression while preserving teacher-level classification capability and explainability.

---

## 🚀 Key Features

* **Dual Knowledge Distillation:** Combines output soft-labels (dark knowledge) with intermediate feature-map alignment.
* **Non-Consecutive Layer Transfer:** Selectively extracts semantic representations from non-adjacent layers (`mixed7` & `mixed10` in InceptionV3; `conv4_block23_out` & `conv5_block3_out` in ResNet101).
* **Dimension Matching Alignment:** Employs Global Average Pooling (GAP), a trainable projection matrix ($W_p$), and $L_2$ normalization to bridge teacher-student feature spatial and channel mismatches.
* **Compact Student Architecture:** Built with depthwise separable convolution layers (`separable_conv2d_3`, `separable_conv2d_6`) to drastically lower parameters and FLOPs.
* **Multi-Objective Loss:** Simultaneously optimizes classification loss, distillation loss, and feature-alignment loss:
  $$\mathcal{L} = \alpha \mathcal{L}_{\text{CE}} + (1 - \alpha) \mathcal{L}_{\text{KD}} + \beta \mathcal{L}_{\text{Feat}}$$

---

## 🏗️ System Architecture
