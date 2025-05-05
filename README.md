# Traffic Sign Classification using Deep Convolutional Neural Networks (CNN)

This project implements a Convolutional Neural Network (CNN) from scratch using TensorFlow/Keras to classify traffic signs. The goal is to build a robust image classification pipeline that can accurately identify various traffic signs from images.


## 📖 Overview

This project demonstrates:
- Preprocessing image data for training deep learning models.
- Building a custom CNN architecture using TensorFlow/Keras.
- Training the model and evaluating its accuracy.
- Saving and loading model checkpoints for later use.

---

## 📂 Dataset

The dataset contains labeled images of traffic signs. Each image represents one of several possible traffic sign categories.

- Number of classes: `43` (if using German Traffic Sign Recognition Benchmark - GTSRB)
- Image size: 32x32 RGB
- Dataset source (if GTSRB): [Kaggle - GTSRB](https://www.kaggle.com/datasets/meowmeowmeowmeowmeow/gtsrb-german-traffic-sign)

You can place the dataset in the `data/` folder or write a script to download and extract it automatically.

---

## 🧠 Model Architecture

The CNN model is composed of:

- 3 Convolutional Layers with ReLU activations
- MaxPooling after each convolution
- Flatten layer
- 2 Dense layers including final Softmax output

The model is compiled with:
- Loss: `categorical_crossentropy`
- Optimizer: `Adam`
- Metrics: `accuracy`

---

## ⚙️ Installation

Clone the repo and install required dependencies:

```bash
git clone https://github.com/YOUR_USERNAME/Traffic-Sign-CNN.git
cd Traffic-Sign-CNN
pip install -r requirements.txt

