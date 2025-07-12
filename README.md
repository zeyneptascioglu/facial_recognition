# Facial Emotion Recognition using PyTorch

This project focuses on predicting human emotional states through facial recognition using deep learning. I developed and trained two convolutional neural network models — a basic ConvNet and a VGG-inspired architecture — to classify facial expressions into seven emotion categories. The project was conducted independently as part of my research exploration in computer vision and affective computing.

## 🎯 Objective

To automatically classify facial expressions into one of the following seven emotions using grayscale images:

- 😠 Angry  
- 🤢 Disgust  
- 😱 Fear  
- 😄 Happy  
- 😢 Sad  
- 😲 Surprise  
- 😐 Neutral

## 🧠 Models

Two different neural network architectures were trained and compared:

- **Basic ConvNet**: A custom-built CNN with multiple convolutional and pooling layers.
- **VGG-inspired Model**: A deeper architecture with batch normalization and more regularization, inspired by the VGG family of networks.

Both models were implemented in PyTorch.

## 📊 Dataset

The [FER-2013](https://www.kaggle.com/datasets/msambare/fer2013) dataset was used:

- 48x48 grayscale facial images.
- 35,887 total images divided into training, validation, and test sets.
- CSV format with pixel values and emotion labels.

The dataset was preprocessed and saved as NumPy arrays for efficient loading.


## 🔧 Features

- CSV parsing and preprocessing to NumPy arrays (`.npy`).
- Torch `Dataset` and `DataLoader` classes for efficient training.
- Training pipeline with checkpoint saving, validation tracking, and loss curve plotting.
- Final evaluation on test data with class-wise accuracy metrics.

## 📈 Results

- Both models were trained for **200 epochs** using **CrossEntropyLoss** and **SGD** optimizer.
- Learning rate scheduling (ExponentialLR) was used to improve convergence.
- The **VGG-based model significantly outperformed** the basic ConvNet on validation and test sets.
- Final results were reported using accuracy metrics for each emotion class.

## 📂 Project Structure
```
├── fer2013.csv # Original dataset
├── train_images.npy # Processed training images
├── train_labels.npy # Processed training labels
├── valid_images.npy # Validation images
├── valid_labels.npy # Validation labels
├── test_images.npy # Test images
├── test_labels.npy # Test labels
├── model.py # Code for model training and evaluation
├── README.md # This file
```

## 🧪 How to Run

1. Download and place `fer2013.csv` in the project directory.
2. Run the preprocessing block to generate `.npy` files.
3. Choose a model: set `model_name = "basic"` or `"vgg"` in the script.
4. Execute the training pipeline.
5. Evaluate the saved model on the test set.

## 📄 Outcome

Based on the experiments, the VGG-based architecture demonstrates better generalization performance and robustness compared to the basic ConvNet. The architecture’s use of batch normalization and dropout significantly helped improve test accuracy and reduce overfitting.

---

*This project was an individual exploration into emotion recognition and deep learning as part of my personal research interests.*
