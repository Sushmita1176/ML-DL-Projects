# Leaf Disease Prediction using Transfer Learning

A binary image classification project that detects whether a plant leaf is **healthy** or **diseased**, using transfer learning with MobileNetV2 and a comparison against classical ML classifiers trained on extracted deep features.

## Overview

This project explores two complementary approaches to leaf disease detection:
1. **End-to-end fine-tuning** of a pretrained MobileNetV2 model
2. **Feature extraction + classical ML** — using MobileNetV2 as a frozen feature extractor, then training traditional classifiers (Logistic Regression, Naive Bayes, Decision Tree) on the extracted features

It also includes an interactive prediction pipeline: upload any leaf image and get an instant Healthy/Diseased prediction with visualized results.

## Dataset

- Sourced from Kaggle (`asheniranga/leaf-disease-dataset-combination`)
- 38,347 total images across 2 classes:
  - **Diseased**: 30,749 images
  - **Healthy**: 7,598 images
- Images resized to 224x224 and organized into class folders for training

## Approach 1: Fine-Tuned MobileNetV2

- Base model: **MobileNetV2** pretrained on ImageNet, with the last 20 layers unfrozen for fine-tuning
- Added: Global Average Pooling → Batch Normalization → Dropout (0.3) → Dense (sigmoid output)
- Optimizer: Adam (learning rate = 1e-4), Loss: Binary Crossentropy
- Trained for 5 epochs

**Results:**
| Epoch | Train Accuracy | Val Accuracy |
|---|---|---|
| 1 | 91.10% | 75.37% |
| 3 | 97.26% | 96.88% |
| 5 | 98.20% | 96.75% |

## Approach 2: Feature Extraction + Classical ML

MobileNetV2 (frozen, `pooling='avg'`) was used to extract 1,280-dimensional feature vectors from every image, which were then used to train and compare traditional classifiers:

| Model | Accuracy |
|---|---|
| **Logistic Regression** | **95.18%** |
| Naive Bayes | 89.45% |
| Decision Tree | (evaluated — see notebook) |

### Logistic Regression — Detailed Metrics

| Class | Precision | Recall | F1-score |
|---|---|---|---|
| Diseased | 0.97 | 0.97 | 0.97 |
| Healthy | 0.87 | 0.89 | 0.88 |

Logistic Regression on extracted features performed nearly as well as full fine-tuning, while being significantly cheaper to train once features are extracted — a useful trade-off when experimenting with multiple classifiers quickly.

## Interactive Prediction Demo

The notebook includes a live inference pipeline:
- Upload any leaf image
- The image is preprocessed and passed through the MobileNetV2 feature extractor
- The scaled features are fed into the trained Logistic Regression model
- Output: a Healthy/Diseased prediction, displayed alongside the uploaded image

## Tech Stack

- Python, TensorFlow / Keras
- MobileNetV2 (transfer learning)
- scikit-learn (Logistic Regression, Naive Bayes, Decision Tree, preprocessing)
- KaggleHub (dataset download)
- Matplotlib, Seaborn

## How to Run

1. Open the notebook in Google Colab
2. Run all cells sequentially — the dataset downloads automatically via `kagglehub`
3. To test the live prediction feature, run the final cells and upload your own leaf image when prompted

## Possible Improvements

- Address class imbalance (Diseased outnumbers Healthy roughly 4:1) using class weighting or oversampling
- Extend beyond binary classification to identify specific disease types, not just presence/absence
- Try additional lightweight backbones (EfficientNet-Lite, MobileNetV3) for comparison
- Deploy the prediction pipeline as a simple web app (e.g., Streamlit or Gradio) for easier demoing
