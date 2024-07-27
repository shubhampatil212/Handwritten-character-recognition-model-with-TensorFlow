# Handwritten Character Recognition with TensorFlow

This project implements a deep learning model for recognizing handwritten words using TensorFlow. The model is trained on the IAM Handwriting Database and can predict text from images of handwritten words.

## Introduction

Handwriting recognition is a challenging task in computer vision and natural language processing. This project aims to create a robust model capable of transcribing handwritten words into digital text. We use a combination of Convolutional Neural Networks (CNNs) and Recurrent Neural Networks (RNNs) to achieve this goal.

Key features of this project:
- Utilizes TensorFlow for model creation and training
- Implements a custom architecture combining CNNs and Bidirectional LSTMs
- Uses CTC (Connectionist Temporal Classification) loss for sequence prediction
- Includes data augmentation techniques to improve model generalization

## Prerequisites

To run this project, you need the following:

- Python 3.7 or higher
- TensorFlow 2.x
- OpenCV
- NumPy
- Pandas
- tqdm

You can install the required packages using: pip install tensorflow opencv-python numpy pandas tqdm
Additionally, you'll need to install the custom `mltu` library, which contains utility functions and classes used in this project.

## Dataset Collection and Preprocessing

This project uses the IAM Handwriting Database, a large database of handwritten English text. The dataset contains forms of unconstrained handwritten text, which were scanned and saved as PNG images with corresponding ground truth text.

### Dataset Structure

The IAM dataset is organized as follows:
- `words/` - Contains word images organized in subdirectories
- `words.txt` - Ground truth file containing information about each word image

### Preprocessing Steps

1. **Download and Extraction**: The script automatically downloads and extracts the dataset if not present.
2. **Data Parsing**: We parse the `words.txt` file to associate each image with its corresponding label.
3. **Image Processing**: Images are resized to a standard size and normalized.
4. **Text Processing**: Labels are encoded using a character set derived from the dataset.

### Data Augmentation

To improve model generalization, we apply the following augmentations during training:
- Random brightness adjustment
- Random rotation
- Random erosion/dilation
- Random sharpening

## Model Architecture

The model uses a combination of:
- Convolutional layers for feature extraction
- Residual connections for better gradient flow
- Bidirectional LSTM layers for sequence processing
- CTC loss for sequence prediction

## Training

The model is trained using the Adam optimizer with a custom learning rate schedule. Early stopping and model checkpointing are used to save the best performing model.

## Inference

After training, the model can be used to predict text from new handwritten word images. The `inferenceModel.py` script demonstrates how to use the trained model for predictions.

## Conclusion

This project demonstrates the application of deep learning techniques to the challenging task of handwriting recognition. By leveraging modern architectures and training techniques, we've created a model capable of transcribing handwritten words with high accuracy.
