
---
# 🚀 Vehicle Image Classifier using TensorFlow CNN & Streamlit

![Python](https://img.shields.io/badge/Python-3.9%2B-blue)
![TensorFlow](https://img.shields.io/badge/Backend-TensorFlow%202.x-orange)
![Streamlit](https://img.shields.io/badge/Frontend-Streamlit-ff4b4b)
![License](https://img.shields.io/badge/License-MIT-yellow.svg)

---

## 📌 Project Overview

This project implements a deep learning pipeline to **classify vehicle images** into three categories: **sedan**, **truck**, and **tractor** using:

- 🧠 A custom-built **Convolutional Neural Network (CNN)** in TensorFlow/Keras
- 🎯 A trained model deployed in a modern **Streamlit** UI
- 📊 Visual confidence bar charts using **Plotly**

It's ideal for:
- Building AI-driven image classification pipelines  
- Demonstrating CNN usage with image data  
- Creating polished Streamlit apps with real-time AI predictions  

---

## Model & App Highlights

### ✅ TensorFlow CNN Features:
- Custom architecture for 64x64 RGB vehicle images  
- Dropout for regularization  
- Softmax output layer for multiclass classification  
- Trained using cross-entropy loss and Adam optimizer  

### ✅ Streamlit App Features:
- Upload `.jpg`, `.jpeg`, or `.png` images  
- View predicted vehicle type with confidence  
- Explore Plotly-powered confidence distribution bar chart  
- See detailed results in a clean, modern UI  

---

## 📁 Project Structure

```plaintext
vehicle_image_classifier/
├── data/
│   ├── images/                    # Vehicle images for training/testing
│   └── vehicle_labels.csv         # Image-label mapping
├── model/
│   ├── vehicle_cnn_model.h5       # Trained CNN model
│   └── label_encoder.pkl          # Fitted LabelEncoder object
├── model_training.py              # CNN training script
├── app.py                         # Streamlit web application
├── requirements.txt               # Python dependencies
├── Dockerfile                     # Optional container deployment
├── .gitignore                     # Git ignore file
└── README.md                      # This documentation
````

---

## How to Use

### Clone the repo

```bash
git clone https://github.com/amitkharche/CV_product_image_categorization_CNN.git
cd CV_product_image_categorization_CNN
```

### Install Dependencies

```bash
pip install -r requirements.txt
```

### Train the Model (if not already trained)

```bash
python model_training.py
```

### Launch the App

```bash
streamlit run app.py
```

---

## 📸 Example Output

Once the app launches:

* Upload a vehicle image
* The AI model returns the **predicted class**
* A **confidence bar chart** shows prediction probabilities for all three classes

```plaintext
📷 Uploaded: tractor_image.png  
🧠 Prediction: TRACTOR  
Confidence: 89.3%
```

---

## 📊 Classes Supported

| Label     | Description                      |
| --------- | -------------------------------- |
| `sedan`   | Standard passenger cars          |
| `truck`   | Commercial vehicles/trucks       |
| `tractor` | Agricultural or utility tractors |

---

## 🐳 Optional: Run in Docker

```bash
docker build -t vehicle-cnn .
docker run -p 8501:8501 vehicle-cnn
```

Then open [http://localhost:8501](http://localhost:8501) in your browser.

---

## 🔍 Tech Stack

* **Python 3.9+**
* **TensorFlow & Keras**
* **Streamlit**
* **Plotly Express**
* **Pandas & PIL**

---

## 🧾 License

Licensed under the [MIT License](LICENSE)

---

## 🤝 Let's Connect

Have questions or ideas for collaboration?

* 💼 [LinkedIn](https://www.linkedin.com/in/amitkharche)
* ✍️ [Medium](https://medium.com/@amitkharche)
* 💻 [GitHub](https://github.com/amitkharche)

---

> ⭐ Star the repo if you find this project useful or educational!
---
