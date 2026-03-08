# 🏥 Pediatric Pneumonia Detection System using ResNet50

[![Python](https://img.shields.io/badge/Python-3.9+-3776AB?style=for-the-badge&logo=python&logoColor=white)](https://www.python.org/)
[![TensorFlow](https://img.shields.io/badge/TensorFlow-2.x-FF6F00?style=for-the-badge&logo=tensorflow&logoColor=white)](https://www.tensorflow.org/)
[![Streamlit](https://img.shields.io/badge/Streamlit-1.x-FF4B4B?style=for-the-badge&logo=streamlit&logoColor=white)](https://streamlit.io/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg?style=for-the-badge)](https://opensource.org/licenses/MIT)

## 📌 Project Overview
This repository contains a web-based prototype for the early detection of **Pediatric Pneumonia** (children aged 1-5 years) using **Deep Learning**. This project is part of a Bachelor's Thesis in **Information Systems** at **Universitas Gunadarma (2026)**.

The system utilizes the **ResNet50** architecture with a **Transfer Learning** approach, trained on a comprehensive dataset of chest X-ray images to provide a clinical decision support tool for medical practitioners.



---

## 🚀 Key Features
* **High Accuracy**: Achieved **87.59% accuracy** on unseen test data.
* **Real-time Analysis**: Instant diagnosis (Normal vs. Pneumonia) after image upload.
* **Confidence Gauge**: Interactive visualization of the AI's confidence level using Plotly.
* **Prescriptive Recommendations**: Provides medical action suggestions based on the prediction results.
* **User-Friendly Interface**: Clean and intuitive UI built with Streamlit for medical environments.

---

## 🛠️ Tech Stack
* **Core Logic**: Python 3.9+
* **Deep Learning**: TensorFlow, Keras (ResNet50)
* **Web Framework**: Streamlit
* **Data Visualization**: Plotly, Matplotlib
* **Image Processing**: Pillow (PIL), NumPy

---

## 📁 Dataset Source
The model was trained using the **Chest X-Ray Images (Pneumonia)** dataset from Kaggle, specifically focusing on the pediatric subset from the Guangzhou Women and Children’s Medical Center.
* **Total Images**: 8,500+
* **Target Group**: Pediatric patients (1-5 years old)

---

## ⚙️ Installation & Usage

### 1. Clone the Repository
```bash
### 1. Clone the Repository
git clone [https://github.com/yogiprasj/Pediatric-Pneumonia-Detection-ResNet50.git](https://github.com/yogiprasj/Pediatric-Pneumonia-Detection-ResNet50.git)
cd Pediatric-Pneumonia-Detection-ResNet50
```

### 2. Install Dependencies
``` bash
pip install -r requirements.txt
```

### 3. Run the Application
```bash
streamlit run app.py
