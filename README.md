# 😷 Face Mask Detection using Deep Learning

<div align="center">

![Python](https://img.shields.io/badge/Python-3.10+-blue?style=for-the-badge&logo=python)
![TensorFlow](https://img.shields.io/badge/TensorFlow-2.x-FF6F00?style=for-the-badge&logo=tensorflow)
![Keras](https://img.shields.io/badge/Keras-Deep%20Learning-D00000?style=for-the-badge&logo=keras)
![Streamlit](https://img.shields.io/badge/Streamlit-Web%20App-FF4B4B?style=for-the-badge&logo=streamlit)
![MobileNetV2](https://img.shields.io/badge/Model-MobileNetV2-success?style=for-the-badge)

### 🚀 AI-powered Face Mask Detection using Transfer Learning and TensorFlow

</div>

---

## 📌 Project Overview

This project is a **Deep Learning-based Face Mask Detection System** built using **TensorFlow, Keras, and MobileNetV2** through **Transfer Learning**.

The application predicts whether a person is wearing a face mask or not from an uploaded image through a professional **Streamlit web application**.

The project demonstrates an end-to-end Computer Vision workflow including:

- Data Loading
- Data Augmentation
- Transfer Learning
- Model Training
- Fine-Tuning
- Model Evaluation
- Streamlit Deployment

---

# 🎯 Problem Statement

Automatically detect whether a person is wearing a face mask from an image.

Classes:

- 😷 With Mask
- ❌ Without Mask

---

# 🖼 Dataset

**Dataset:** Face Mask 12K Images Dataset

Dataset Structure

```
Train/
    WithMask/
    WithoutMask/

Validation/
    WithMask/
    WithoutMask/

Test/
    WithMask/
    WithoutMask/
```

Dataset Statistics

| Split | Images |
|--------|--------:|
| Train | 10,000 |
| Validation | 800 |
| Test | 992 |

---

# 🧠 Model Architecture

Transfer Learning Model

```
Input Image (224×224×3)
        │
        ▼
Data Augmentation
        │
        ▼
Rescaling
        │
        ▼
MobileNetV2 (Pretrained on ImageNet)
        │
        ▼
GlobalAveragePooling2D
        │
        ▼
Dropout(0.2)
        │
        ▼
Dense(1, Sigmoid)
        │
        ▼
Prediction
```

---

# ⚙ Technologies Used

- Python
- TensorFlow
- Keras
- MobileNetV2
- NumPy
- Matplotlib
- Scikit-Learn
- PIL
- Streamlit

---

# 📈 Training Pipeline

- Data Augmentation
- Image Rescaling
- MobileNetV2 Feature Extractor
- Binary Crossentropy Loss
- Adam Optimizer
- EarlyStopping
- ModelCheckpoint
- Fine-Tuning

---

# 📊 Model Evaluation

Evaluation techniques used:

- ✅ Accuracy
- ✅ Loss Curve
- ✅ Confusion Matrix
- ✅ Classification Report
- ✅ Random Predictions
- ✅ Test Dataset Evaluation

---

# 📷 Sample Results

## Accuracy Plot

> Replace with your image

```
accuracy_plot.png
```

---

## Loss Plot

> Replace with your image

```
loss_plot.png
```

---

## Confusion Matrix

> Replace with your image

```
confusion_matrix.png
```

---

# 🚀 Streamlit Web Application

Features:

- Upload Image
- Image Preview
- Real-Time Prediction
- Confidence Score
- Professional UI
- Error Handling
- Responsive Layout

---

# 📂 Project Structure

```
Face-Mask-Detection/
│
├── Face_Mask_Detection.ipynb
├── app.py
├── requirements.txt
├── README.md
├── face_mask_detector.keras
├── accuracy_plot.png
├── loss_plot.png
├── confusion_matrix.png
└── sample_predictions/
```

---

# ▶ How to Run

### Clone Repository

```bash
git clone https://github.com/YOUR_USERNAME/Face-Mask-Detection.git
```

### Move into Project

```bash
cd Face-Mask-Detection
```

### Install Dependencies

```bash
pip install -r requirements.txt
```

### Run Streamlit

```bash
streamlit run app.py
```

---

# 🎯 Future Improvements

- 🎥 Real-time Webcam Detection
- 📱 Mobile Deployment
- ☁ Cloud Deployment
- 😷 Multi-Class Mask Detection
- 🚀 TensorFlow Lite Conversion

---

# 💡 Skills Demonstrated

- Deep Learning
- Computer Vision
- Transfer Learning
- TensorFlow
- Keras
- MobileNetV2
- Streamlit
- Model Deployment
- Data Augmentation
- Binary Classification

---

# 📬 Connect With Me

**GitHub**

https://github.com/YOUR_USERNAME

**LinkedIn**

https://linkedin.com/in/YOUR_LINKEDIN

---

# ⭐ If you found this project useful, don't forget to Star the repository!
