<h1 align="center">FashionMNIST Classification with PyTorch</h1>

<h3>Overview</h3>

This project implements two neural network architectures (CNN and ResNet) to classify images from the FashionMNIST dataset using PyTorch. The project consists of four main scripts:
1. Loading-Preprocessing-Training-Saving.py: Handles data loading, preprocessing, model training, and saving the trained model.
2. Loading-Evaluating-ConfusionMatrix.py: Loads the trained model, evaluates its performance on the test set, and generates a confusion matrix.
3. ResNet-18.py: Implements, trains, and saves a ResNet-style model for FashionMNIST classification.
4. ResNet-18_Evaluation.py: Loads and evaluates the trained ResNet model with a confusion matrix.

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

<h5>CNN Model</h5>

  1. Training the cnn-model
  Run the training script : `python Loading-Preprocessing-Training-Saving.py`

  This will :
  - Download the FashionMNIST dataset
  - Preprocess the data
  - Train the CNN model for 4 epochs
  - Save the trained model to `models/CNN-Model.pt`<br>

  2. Evaluating the cnn-model
  Run the evaluation script : `python Loading-Evaluating-ConfusionMatrix.py`

  This will :
  - Load the trained model
  - Evaluate the model on the test set
  - Display test loss and accuracy
  - Generate and display a confusion matrix

<h5>ResNet Model</h5>

  1. Training the resnet-model
  Run the training script : `python ResNet-18.py`

  This will :
  - Download the FashionMNIST dataset
  - Preprocess the data
  - Train the ResNet model for 4 epochs
  - Save the trained model to `models/ResNet-Model.pt`<br>

  2. Evaluating the resnet-model
  Run the evaluation script : `python ResNet-18_Evaluation.py`

  This will :
  - Load the trained model
  - Evaluate the model on the test set
  - Display test loss and accuracy
  - Generate and display a confusion matrix

<h3>Model Architecture</h3>

<h5>CNN Model</h5>

  The CNN model consists of :
  - Two convolutional blocks, each containing:
    - Two convolutional layers with ReLU activation
    - One max pooling layer
  
  - A classifier with :
    - A flattening layer
    - A linear output layer

<h5>ResNet Model</h5>

  The ResNet-style model features :
  - An initial convolutional block with:
    - Convolutional layer
    - Batch normalization
    - ReLU activation
    - Max pooling
  
  - Two residual-style layers with:
    - Convolutional layers
    - Batch normalization
    - ReLU activation
  
  - A final classification layer with:
    - Adaptive average pooling
    - Flattening
    - Linear output layer

<h3>Results</h3>

After training for 4 epochs, the model typically achieves :
- CNN Model:
  - Training accuracy: ~85-90%
  - Test accuracy: ~85-88%
- ResNet Model:
  - Training accuracy: ~85-92%
  - Test accuracy: ~86-90%
The confusion matrix provides detailed insights into which classes the model confuses most often.

<h3>Customizations</h3>

You can modify various parameters in the scripts :
- `BATCH_SIZE` in both scripts
- Model architecture in `CNN_Model` class
- Training parameters (epochs, learning rate) in the training script
- Hidden units and other model hyperparameters

<h3>Comparison</h3>

The project allows for direct comparison between:
1. A traditional CNN architecture
2. A ResNet-inspired architecture

Both approaches are implemented with similar training regimes, enabling fair comparison of their performance on the FashionMNIST dataset.
