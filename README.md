# Image Forgery Detection

An Explainable Deep Learning Approach Using EfficientNetB0 and Grad-CAM

## 1. Project Overview

Image Forgery Detection is a deep learning-based project that identifies whether a digital image is Real or Forged.

The project uses EfficientNetB0 with transfer learning for image classification and Grad-CAM (Gradient-weighted Class Activation Mapping) to visualize the image regions that influence the model's predictions.

An interactive web interface is developed using Streamlit, allowing users to upload images and view classification results with visual explanations.

## 2. Problem Statement

With the increasing use of digital image editing tools, manipulated images are becoming difficult to distinguish from authentic images.

This project aims to develop an automated image classification system that can identify potentially forged images using deep learning techniques and provide visual explanations of its predictions.

## 3. Objectives

- Develop a deep learning model to classify images as Real or Forged.
- Implement EfficientNetB0 using transfer learning.
- Apply image preprocessing techniques.
- Integrate Grad-CAM for model explainability.
- Develop an interactive Streamlit application.
- Display prediction results and heatmap visualizations.

## 4. Technologies Used

| Technology | Purpose |
|---|---|
| Python | Programming language |
| TensorFlow | Deep learning framework |
| Keras | Model development |
| EfficientNetB0 | Image feature extraction |
| Grad-CAM | Explainable AI |
| NumPy | Numerical operations |
| Pillow | Image processing |
| Streamlit | Web application |

## 5. Dataset

**Dataset:** MICC-F2000

The MICC-F2000 dataset is used for image forgery detection research. It contains original and tampered images, including copy-move forgery examples.

The project uses original and forged image categories for model development and evaluation.

**Note:** Dataset images are not included in this repository. Users must obtain the dataset from an authorized source and arrange the images according to the directory structure expected by the project.

## 6. Model Architecture

The project uses EfficientNetB0 with transfer learning.

The classification pipeline consists of:

1. Input image resizing to 256 × 256 pixels.
2. EfficientNetB0 feature extraction.
3. Global Average Pooling.
4. Dropout layer.
5. Dense layer with ReLU activation.
6. Final sigmoid output layer for binary classification.

The model produces a prediction for one of two classes:

- Real
- Forged

## 7. Grad-CAM Explainability

Grad-CAM is used to generate heatmaps that highlight image regions influencing the model's prediction.

It helps users understand which areas contributed to the classification result.

**Note:** Grad-CAM provides a visual explanation of model predictions. It does not generate an exact pixel-level forgery mask.

## 8. Project Structure

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

## 9. Installation and Setup

### Step 1: Clone the repository

```bash
git clone https://github.com/Bhuvana6303/Explainable-Image-Forgery-Detection.git
```

### Step 2: Navigate to the project directory

```bash
cd Explainable-Image-Forgery-Detection
```

### Step 3: Create a virtual environment

```bash
python -m venv .venv
```

### Step 4: Activate the virtual environment

For Windows PowerShell:

```powershell
.\.venv\Scripts\Activate.ps1
```

### Step 5: Install dependencies

```bash
pip install -r requirements.txt
```

### Step 6: Run the Streamlit application

```bash
python -m streamlit run web/app.py
```

Open the local URL displayed in the terminal, usually:

`http://localhost:8501`

## 10. Application Features

- Upload digital images.
- Classify images as Real or Forged.
- Display prediction results.
- Generate Grad-CAM heatmaps.
- Interactive Streamlit interface.

## 11. Future Enhancements

- Improve model performance using larger and more diverse datasets.
- Explore additional deep learning architectures.
- Develop more precise forgery localization techniques.
- Deploy the application on a cloud platform.
- Improve explainability and visualization techniques.

## 12. Author

**Bhuvana Gundepalli**

B.Tech – Computer Science and Engineering

## 13. GitHub Repository

[Image Forgery Detection – GitHub](https://github.com/Bhuvana6303/Explainable-Image-Forgery-Detection)
