# CIFAR-10 Image Classification with CNN-LeNet

A convolutional neural network built with TensorFlow/Keras to classify images from the CIFAR-10 dataset into 10 categories (airplane, automobile, bird, cat, deer, dog, frog, horse, ship, truck).

## Overview

This project implements a classic **LeNet-5 style architecture** on the CIFAR-10 dataset, with data augmentation and adaptive learning rate scheduling to improve generalization on a relatively small, low-resolution (32x32) image dataset.

## Dataset

- **CIFAR-10**: 60,000 32x32 color images across 10 classes
- Split into training (40,000), validation (10,000), and test (10,000) sets using stratified sampling
- Pixel values normalized to the [0, 1] range
- Labels one-hot encoded

## Data Augmentation

To improve model robustness and reduce overfitting, training images were augmented using:
- Random rotation (up to 15 degrees)
- Random horizontal flips

## Model Architecture

A LeNet-inspired CNN:

```
Conv2D(6 filters, 5x5, tanh) 
    -> AveragePooling2D(2x2)
    -> Conv2D(16 filters, 5x5, tanh)
    -> AveragePooling2D(2x2)
    -> Flatten
    -> Dense(120, tanh)
    -> Dense(84, tanh)
    -> Dense(10, softmax)
```

## Training Setup

- **Optimizer**: Adam (learning rate = 0.001)
- **Loss**: Categorical Crossentropy
- **Callbacks**:
  - `EarlyStopping` (monitors validation loss, patience = 5, restores best weights)
  - `ReduceLROnPlateau` (halves learning rate on plateau, patience = 2)
- Trained for up to 50 epochs (stopped early at epoch 36 via EarlyStopping)

## Results

| Metric | Value |
|---|---|
| Total Epochs Run | 36 |
| Final Training Accuracy | 66.31% |
| Best Validation Accuracy | ~59.01% |
| Final Training Loss | ~0.96 |
| Best Validation Loss | ~1.18 |

Model performance was further evaluated using a **confusion matrix** on the test set, which showed strong performance on distinct classes like automobile, ship, and frog, with more confusion between visually similar animal classes (cat/dog, deer/horse).

Training and validation accuracy/loss curves were plotted to visualize convergence behavior across epochs.

## Tech Stack

- Python
- TensorFlow / Keras
- NumPy, Pandas
- Matplotlib, Seaborn
- scikit-learn (train/test split, confusion matrix)

## How to Run

1. Open the notebook in Google Colab or Jupyter
2. Run all cells sequentially — the CIFAR-10 dataset downloads automatically via `tensorflow.keras.datasets`
3. No additional setup or API keys required

## Possible Improvements

- Replace `tanh` activations with `ReLU` for faster convergence
- Add `Dropout` and `BatchNormalization` layers (imported but unused in current architecture) to reduce overfitting
- Experiment with deeper architectures (e.g., ResNet-style blocks) to push validation accuracy higher
- Try `RMSprop` (imported but unused) as an alternative optimizer for comparison
