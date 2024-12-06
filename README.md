# Industrial Defect Detection using Deep Learning

## Project Overview
This project implements a deep learning solution for detecting defects in industrial components using computer vision. The system combines two distinct datasets and employs data augmentation techniques to create a robust defect detection model.

## Dataset Description

### Dataset Sources
1. Industrial Tools Classification Dataset (Smaller Dataset)
   - Source: Kaggle (niharikaamritkar/industrial-tools-classification)
   - Contains images of industrial tools with defective and non-defective classifications
   - Initially smaller in size, enhanced through augmentation

2. Casting Product Dataset (Larger Dataset)
   - Source: Kaggle (ravirajsinh45/real-life-industrial-dataset-of-casting-product)
   - Comprises casting product images with defective ("def_front") and non-defective ("ok_front") samples
   - Larger dataset used to complement the first dataset

## Data Augmentation Strategy

### Augmentation Techniques
The smaller dataset was augmented using the following transformations:
- Random Horizontal Flip (50% probability)
- Random Vertical Flip (50% probability)
- Random Rotation (up to 30 degrees)
- Color Jitter
  - Brightness adjustment: ±20%
  - Contrast adjustment: ±20%
  - Saturation adjustment: ±20%
  - Hue adjustment: ±10%
- Random Affine Transformation
  - Rotation: up to 20 degrees
  - Scale: 80% to 120%

For each original image, 5 augmented copies were generated, significantly increasing the dataset size and introducing valuable variations.

## Model Architecture and Training

### Model Details
- Base Architecture: ResNet-18 (pretrained)
- Modifications:
  - Adapted final fully connected layer for binary classification
  - Input size: 224x224 pixels
  - Output classes: 2 (defective, non-defective)

## Model Performance

### Training Results
- Final Training Accuracy: 96.42%
- Final Training Loss: 0.1106
- Final Validation Accuracy: 90.26%
- Final Validation Loss: 0.2060

### Test Set Performance
- Overall Accuracy: 92%

#### Class-wise Performance:
- Defective Class:
  - Precision: 0.89
  - Recall: 0.96
  - F1-score: 0.92
  
- Non-defective Class:
  - Precision: 0.96
  - Recall: 0.88
  - F1-score: 0.91

### Confusion Matrix Analysis
```
              Predicted Defective    Predicted Non-defective
Actual Defective         603                 26
Actual Non-defective     78                  557
```

The model shows strong performance with balanced precision and recall across both classes, achieving an overall accuracy of 92%. The confusion matrix indicates a slightly higher tendency to correctly identify defective items (603 true positives) compared to non-defective items (557 true negatives).

## Project Structure
```
C:.
│   Defect_Detection.ipynb    # Main notebook for development and testing
│   model_best.pth           # Saved model weights
│   README.md                # Project documentation
│   requirement.txt          # Dependencies
│
└───src
        data_preparation.py  # Dataset processing and augmentation
        evaluate.py         # Model evaluation scripts
        model.py           # Model architecture definition
        train.py           # Training pipeline
```

## Dependencies
- PyTorch
- torchvision
- PIL
- scikit-learn
- matplotlib
- kagglehub
