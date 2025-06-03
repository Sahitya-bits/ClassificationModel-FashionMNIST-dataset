<h1 align="center">FashionMNIST Classification with PyTorch</h1>

<h3>Overview</h3>

This project implements a Convolutional Neural Network (CNN) to classify images from the FashionMNIST dataset using PyTorch. The project consists of two main scripts:
1. Loading-Preprocessing-Training-Saving.py: Handles data loading, preprocessing, model training, and saving the trained model.
2. Loading-Evaluating-ConfusionMatrix.py: Loads the trained model, evaluates its performance on the test set, and generates a confusion matrix.

<h3>Requirments</h3>

- Python 3.x<br>
- PyTorch<br>
- torchvision<br>
- torchmetrics<br>
- mlxtend<br>
- matplotlib<br>
- tqdm<br>
- numpy<br>

Install the required packages with : `pip install torch torchvision torchmetrics mlxtend matplotlib tqdm numpy`

<h3>Usage</h3>

1. Training the model
Run the training script : `python Loading-Preprocessing-Training-Saving.py`<br>
This will :
- Download the FashionMNIST dataset
- Preprocess the data
- Train the CNN model for 4 epochs
- Save the trained model to `models/CNN-Model.pt`<br>

2. Evaluating the model
Run the evaluation script : `python Loading-Evaluating-ConfusionMatrix.py`<br>
This will :
- Load the trained model
- Evaluate the model on the test set
- Display test loss and accuracy
- Generate and display a confusion matrix

<h3>Model Architecture</h3>

The CNN model consists of :
- Two convolutional blocks, each containing:
  - Two convolutional layers with ReLU activation
  - One max pooling layer

- A classifier with :
  - A flattening layer
  - A linear output layer

<h3>Results</h3>

After training for 4 epochs, the model typically achieves :
- Training accuracy: ~85-90%
- Test accuracy: ~85-88%
The confusion matrix provides detailed insights into which classes the model confuses most often.

<h3>Customizations</h3>

You can modify various parameters in the scripts :
- `BATCH_SIZE` in both scripts
- Model architecture in `CNN_Model` class
- Training parameters (epochs, learning rate) in the training script
