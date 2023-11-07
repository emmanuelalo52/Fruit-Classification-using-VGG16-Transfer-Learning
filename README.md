# Fruit Classification using VGG16 Transfer Learning

This repository contains a Python script for classifying fruits using a pre-trained VGG16 model with the Keras library and TensorFlow backend. The code demonstrates transfer learning, where the VGG16 model's convolutional layers are used as feature extractors, and custom dense layers are added for classification.

## Purpose

The purpose of this code is to create a fruit classification model using a pre-trained VGG16 model, transfer learning, and custom layers for fine-tuning. The code includes data preprocessing, model creation, training, and evaluation using confusion matrices and performance visualizations.

## Dependencies

Before using this code, make sure you have the following dependencies installed:

- TensorFlow
- Keras
- NumPy
- Matplotlib
- Scikit-Learn

You can install these libraries using pip:

```shell
pip install tensorflow keras numpy matplotlib scikit-learn
```

## Dataset

The code uses a fruit dataset containing images of various fruits for training and testing. The dataset is organized into subfolders, each containing images of a specific fruit.

## Usage

1. Set the `train_path` and `test_path` variables to the paths of your training and testing datasets.
2. Configure the image size, number of epochs, and batch size as needed.
3. Preprocess the data and create an instance of the VGG16 model without the top classification layer.
4. Add custom classification layers to the model.
5. Compile the model with the specified loss function and optimizer.
6. Create an image data generator for data augmentation.
7. Train the model using the training and validation datasets.
8. Evaluate the model's performance and generate confusion matrices.
9. Visualize the loss and accuracy during training.

## Code Structure

- `vgg`: A VGG16 model with the top classification layer removed.
- `model`: A custom classification model built on top of the VGG16 model.
- `train_generator` and `test_generator`: Data generators for training and testing data.
- `fit_model`: Trains the model on the training data and validates it on the testing data.
- `get_confusion_matrix`: Generates confusion matrices for the training and testing datasets.
- `plot_confusion_matrix`: Utility function for visualizing confusion matrices.

## Acknowledgments

This code is adapted from various sources and examples in the Keras and TensorFlow documentation, as well as from existing tutorials and projects related to transfer learning and image classification.

Feel free to use, modify, and extend this code for your own image classification tasks or computer vision projects. Enjoy classifying fruits with VGG16!
