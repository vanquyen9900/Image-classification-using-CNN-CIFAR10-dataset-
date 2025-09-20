# 🚀 Project Overview

This project demonstrates image classification using **Convolutional Neural Networks (CNNs)** on the **CIFAR-10 dataset**. The notebook compares performance between a basic Artificial Neural Network (ANN) and a CNN, highlighting the advantages of convolutional layers for image-related tasks.

---

## 📊 Dataset

The **CIFAR-10 dataset** contains 60,000 32x32 color images across 10 classes:

- ✈️ airplane
- 🚗 automobile
- 🐦 bird
- 🐱 cat
- 🦌 deer
- 🐶 dog
- 🐸 frog
- 🐴 horse
- 🚢 ship
- 🚚 truck

- Training set: 50,000 images
- Test set: 10,000 images

---

## ⚙️ Workflow

1. 📥 **Library Imports**: Load TensorFlow, Keras, NumPy, Matplotlib.
2. 📂 **Data Loading**: CIFAR-10 dataset is loaded via Keras.
3. 🧹 **Data Preprocessing**: Normalization of pixel values.
4. 🧠 **Model Building**:
   - ANN: Flatten → Dense layers.
   - CNN: Conv2D + MaxPooling → Flatten → Dense layers.
5. 🎯 **Model Training**: Fit ANN and CNN separately.
6. 📉 **Evaluation**: Compare accuracy and loss on test set.
7. 📊 **Visualization**: Show training performance, predictions, and sample results.

---

## 🛠️ Requirements

- Python 3.x
- TensorFlow / Keras
- NumPy
- Matplotlib

Install dependencies:

```bash
pip install tensorflow numpy matplotlib
```

---

## ▶️ Usage

1. Clone the repository.
2. Open the notebook:

```bash
jupyter notebook "Image classification using CNN (CIFAR10 dataset).ipynb"
```

3. Run cells step by step to:
   - Load dataset
   - Train models
   - Evaluate performance

---

## 📈 Results

- ANN achieved lower accuracy due to its inability to capture spatial features.
- CNN significantly outperformed ANN with higher accuracy and better generalization.
- Visualizations clearly show improved predictions by CNN.

---

## 🔮 Future Improvements

- Add **data augmentation** for better generalization.
- Explore **deeper CNN architectures** (ResNet, VGG, etc.).
- Apply **transfer learning** with pretrained models.
- Implement **regularization** techniques to reduce overfitting.

---

## 📚 References

- [CIFAR-10 Dataset](https://www.cs.toronto.edu/~kriz/cifar.html)
- [Keras Documentation](https://keras.io/)
- Deep Learning with Python, François Chollet
