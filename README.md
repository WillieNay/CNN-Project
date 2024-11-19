# MNIST Handwritten Digit Classification

This project implements a Convolutional Neural Network (CNN) to classify handwritten digits using the MNIST dataset. The trained model is saved and used for real-time predictions on new digit images.

## Table of Contents

- [Overview](#overview)
- [Technologies Used](#technologies-used)
- [Dataset](#dataset)
- [Model Architecture](#model-architecture)
- [Installation](#installation)
- [Usage](#usage)
- [Example Output](#example-output)
- [Contributing](#contributing)
- [License](#license)

---

## Overview

The MNIST dataset is a collection of 70,000 grayscale images of handwritten digits (0-9) with a resolution of 28x28 pixels. This project trains a deep learning model to achieve high accuracy in digit classification and demonstrates its usage for predicting new handwritten digits.

## Technologies Used

- Python 3.x
- TensorFlow / Keras
- NumPy
- Matplotlib
- OpenCV
- PIL (Pillow)

---

## Dataset

The MNIST dataset is directly loaded using `keras.datasets.mnist`. It is pre-split into training and testing datasets:
- **Training Set**: 60,000 images
- **Testing Set**: 10,000 images

---

## Model Architecture

The Convolutional Neural Network (CNN) used has the following structure:
1. **Input Layer**: Accepts 28x28 grayscale images.
2. **Convolutional Layer 1**: 32 filters, kernel size = 5x5, ReLU activation, "same" padding.
3. **MaxPooling Layer 1**: Pool size = 2x2, "same" padding.
4. **Convolutional Layer 2**: 64 filters, kernel size = 5x5, ReLU activation, "same" padding.
5. **MaxPooling Layer 2**: Pool size = 2x2, "same" padding.
6. **Flatten Layer**: Flattens the feature map into a vector.
7. **Dense Layer 1**: 1024 units, ReLU activation, Dropout = 0.2.
8. **Dense Layer 2 (Output)**: 10 units (one for each digit), sigmoid activation.

---
