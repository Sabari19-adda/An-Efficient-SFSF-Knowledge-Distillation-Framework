# An Efficient Selective Feature-Layer and Soft-Label Fused Knowledge Distillation Framework for Building a
Lightweight Multi-Stage Alzheimer’s Disease Classification System

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![TensorFlow 2.x](https://img.shields.io/badge/TensorFlow-2.x-orange.svg)](https://tensorflow.org/)

An end-to-end framework for transferring hierarchical knowledge from high-capacity teacher models (InceptionV3 / ResNet101) to an ultra-lightweight student neural network using non-consecutive feature layer alignment and soft-label supervision for multi-stage Alzheimer's Disease MRI classification.

---

## 📌 Abstract

Deploying deep learning models in resource-constrained medical environments is hindered by massive memory footprints and heavy computational demands. **SFSF-KD** resolves this by fusing soft-label probability distributions (via temperature-scaled KL divergence) with selective intermediate and deep semantic feature transfer from non-consecutive teacher layers. The framework achieves high-ratio model compression while preserving teacher-level classification capability and explainability.

---

## 💻 Tech Stack & Tools

* **Languages:** Python 3.8+
* **Deep Learning:** TensorFlow 2.x / Keras
* **Computer Vision & Analysis:** OpenCV, NumPy, SciPy, Scikit-learn, Matplotlib
* **Explainability:** Grad-CAM
* **Environments:** Google Colab / Kaggle

---

📜 License
This project is licensed under the MIT License - see the LICENSE file for details.
