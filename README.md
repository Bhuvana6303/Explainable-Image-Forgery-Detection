# Image Forgery Detection

## Project Overview

Image Forgery Detection is a deep learning-based application designed to classify digital images as **Real** or **Forged**. The project uses EfficientNetB0 with transfer learning for image classification and Grad-CAM (Gradient-weighted Class Activation Mapping) to visualize the regions that influence the model's predictions.

The application provides an interactive interface using Streamlit, allowing users to upload images and view prediction results along with Grad-CAM visualizations.

## Problem Statement

Digital image manipulation has become increasingly common, making it difficult to distinguish authentic images from forged images. This project aims to develop an automated image classification system that helps identify potentially manipulated images using deep learning techniques.

## Objectives

- Develop a deep learning model to classify images as Real or Forged.
- Implement EfficientNetB0 using transfer learning.
- Preprocess images for model prediction.
- Apply Grad-CAM to visualize important image regions influencing predictions.
- Develop an interactive web application using Streamlit.

## Technologies Used

- **Programming Language:** Python
- **Deep Learning Framework:** TensorFlow and Keras
- **Pre-trained Model:** EfficientNetB0
- **Explainable AI:** Grad-CAM
- **Web Framework:** Streamlit
- **Image Processing:** Pillow, NumPy
- **Development Environment:** VS Code

## Dataset

**Dataset Name:** MICC-F2000

The MICC-F2000 dataset is used for image forgery detection research and contains original and tampered images.

The project uses original and forged image categories for model development and evaluation.

Note: The dataset files are not included in this GitHub repository. Obtain the dataset from its authorized source and arrange the images according to the folder structure expected by the project.

## Model Architecture

The project uses EfficientNetB0 as a feature extraction backbone.

The classification architecture includes:

1. EfficientNetB0 for feature extraction.
2. Global Average Pooling layer.
3. Dropout layer.
4. Dense layer with ReLU activation.
5. Final sigmoid output layer for binary classification.

The model predicts whether an input image belongs to the Real or Forged class.

## Grad-CAM Explainability

Grad-CAM is used to generate heatmaps that highlight image regions contributing to the model's prediction.

It helps users understand which areas influenced the classification result.

**Note:** Grad-CAM provides visual explanations of model predictions. It does not generate an exact pixel-level forgery mask.

## Project Structure

```text
Explainable-Image-Forgery-Detection/
│
├── api/
│   └── app.py
│
├── src/
│   ├── inference.py
│   ├── model.py
│   ├── preprocess.py
│   ├── test.py
│   ├── train.py
│   └── utils.py
│
├── web/
│   └── app.py
│
├── forgery_classifier.h5
├── main.ipynb
├── requirements.txt
├── README.md
└── .gitignore
```

## Installation and Setup

### 1. Clone the repository

```bash
git clone https://github.com/Bhuvana6303/Explainable-Image-Forgery-Detection.git
```

### 2. Navigate to the project directory

```bash
cd Explainable-Image-Forgery-Detection
```

### 3. Create a virtual environment

```bash
python -m venv .venv
```

### 4. Activate the virtual environment

Windows PowerShell:

```powershell
.\.venv\Scripts\Activate.ps1
```

### 5. Install dependencies

```bash
pip install -r requirements.txt
```

### 6. Run the Streamlit application

```bash
python -m streamlit run web/app.py
```

Open the local URL displayed in the terminal, usually:

```text
http://localhost:8501
```

## Application Features

- Upload digital images.
- Predict Real or Forged image categories.
- Display classification results.
- Generate Grad-CAM heatmap visualizations.
- Interactive Streamlit interface.

## Future Enhancements

- Improve model performance using a larger and more diverse dataset.
- Explore additional deep learning architectures.
- Develop more precise image manipulation localization.
- Deploy the application on a cloud platform.
- Enhance model explainability.

## Author

**Bhuvana Gundepalli**

B.Tech – Computer Science and Engineering

## GitHub Repository

https://github.com/Bhuvana6303/Explainable-Image-Forgery-Detection
