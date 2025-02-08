# Image Classification Using Tensorflow
# COMPANY : CODTECH IT SOLUTIONS
# NAME : BANSODE POOJA
# INTERN ID : CT6WJOA
# DOMAIN : MACHINE LEARNING
# DURATION : 6 WEEKS
# MENTOR : NEELA SANTHOSH KUMAR

## This project focuses on building a **Convolutional Neural Network (CNN)** for classifying images into 6 categories: `buildings`, `forest`, `glacier`, `mountain`, `sea`, and `street`. 
The dataset contains around 25k images of size 150x150, sourced from [Kaggle](https://www.kaggle.com/datasets/puneet6060/intel-image-classification).

## Project Overview
The goal of this project is to build and train a CNN model using **TensorFlow** or **PyTorch** to classify images into one of the 6 categories. The project includes:
- Data loading and preprocessing.
- Building and training a CNN model.
- Evaluating the model on a test dataset.
- Saving the trained model for future use.
- Making predictions on new images.

## Dataset
The dataset is available on [Kaggle](https://www.kaggle.com/datasets/puneet6060/intel-image-classification). It contains the following folders:
- `seg_train/`: Training images (6 classes, ~14k images).
- `seg_test/`: Test images (6 classes, ~3k images).

Each class has its own subfolder:
- `buildings/`
- `forest/`
- `glacier/`
- `mountain/`
- `sea/`
- `street/`

## Results
After training, the model achieves the following performance:
Test Accuracy: ~85% (may vary depending on hyperparameters and training time).
