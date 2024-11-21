# ANN MLP Greek Letters Backend

## Description
This project implements a machine learning solution for recognizing lowercase Greek letters using a custom Neural Network (MLP) with Histogram of Oriented Gradients (HOG) feature extraction. The backend provides an API for preprocessing images, extracting features, and predicting Greek letters with high accuracy.

The neural network is designed from scratch using NumPy and leverages computer vision techniques to preprocess and analyze handwritten Greek letter images. It offers a comprehensive pipeline from image processing to letter recognition.

## Table of Contents
 - [Description](#description)
 - [Table of Contents](#table-of-contents)
 - [Features](#features)
 - [Technical Details](#technical-details)
 - [Installation](#installation)
 - [Usage](#usage)
 - [Project Structure](#project-structure)
 - [Neural Network Architecture](#neural-network-architecture)
 - [API Endpoints](#api-endpoints)
 - [Contributing](#contributing)
 - [Frontend Repository](#frontend-repository)


## Features
- Custom Neural Network implementation from scratch
- HOG (Histogram of Oriented Gradients) feature extraction
- Image preprocessing pipeline
- FastAPI-based prediction endpoint
- Model training, evaluation, and persistence
- Interactive CLI for model management
- Confusion matrix visualization
- Confidence-based predictions

## Technical Details
- Language: Python
- Machine Learning: Custom Neural Network (Multilayer Perceptron)
- Feature Extraction: HOG
- Image Processing: OpenCV
- Web Framework: FastAPI
- Visualization: Matplotlib, Seaborn

## Installation
1. Clone the repository:
```bash
  git clone https://github.com/your-username/greek-letters-neural-network.git
  cd greek-letters-neural-network
```
  
## Create a virtual environment (optional but recommended):
```bash
  python -m venv venv
  venv\Scripts\activate  # On MacOS, use `source venv/bin/activate`
```

## Install dependencies:
```bash
pip install -r requirements.txt
```

## Usage
Running the Application
```bash
python greek_letter_recognition.py
```
## Main Menu Options
1. Training Mode
   - Load and preprocess images
   - Extract HOG features
   - Train neural network
   - Evaluate model
   - Save trained model

2. Testing Mode
   - Load pre-trained model
   - Test with individual images

3. Run API Server
   - Start FastAPI server for predictions


## Project Structure
```mk
ia1114_RNA_GreekLetters/
│
├── main.py   # Main application script
├── greek_letters_model.pkl       # Trained model (name by default once generated)
├── requirements.txt               # Project dependencies
└── Greek_Letters/                # Training image dataset
    ├── alpha/
    ├── beta/
    └── ...
```

## Neural Network Architecture
- Input Layer: 3780 HOG features
- Hidden Layer: 392 neurons
- Output Layer: 24 classes (Greek letters)
- Activation Function: Hyperbolic Tangent (tanh)
- Output Activation: Softmax
- Training Algorithm: Backpropagation

## API Endpoints

- GET /: Health check
- POST /predict: Predict Greek letter from uploaded image .png


## Contributing

- Fork the repository
- Create your feature branch (git checkout -b feature/AmazingFeature)
- Commit your changes (git commit -m 'Add some AmazingFeature')
- Push to the branch (git push origin feature/AmazingFeature)
- Open a Pull Request


## Frontend Repository

You can access the frontend repository [here](https://github.com/CodeBreaker518/ANN_MLP_GreekLetters_Frontend.git)

