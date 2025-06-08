# 🚗 Vehicle Classification from CCTV (Smart City)

## 📌 Business Use Case

In smart city infrastructure, classifying vehicle types from surveillance cameras helps:
- Monitor traffic flow by vehicle category
- Analyze parking patterns
- Trigger alerts for restricted zones
- Plan urban development and environmental impact studies

This project uses a CNN to classify images of vehicles into categories like sedan, SUV, truck, bus, and motorbike from simulated CCTV footage.

## 💼 Model Features

- Image preprocessing (64x64 resize, normalization)
- CNN architecture with dropout
- Label encoding and prediction probability distribution
- Deployment with Streamlit

## 📁 Project Structure

```
vehicle_classification_project/
├── data/
│   ├── images/                  # Simulated vehicle images
│   └── vehicle_labels.csv       # Metadata
├── model/                       # Trained model + encoder
├── model_training.py            # CNN training script
├── app.py                       # Streamlit web app
├── requirements.txt             # Python dependencies
├── Dockerfile                   # Containerization
├── .gitignore                   # Ignore untracked files
├── .gitattributes               # Git attributes
└── README.md                    # Project documentation
```

## 🚀 How to Run

### Install dependencies
```bash
pip install -r requirements.txt
```

### Train the model
```bash
python model_training.py
```

### Launch the app
```bash
streamlit run app.py
```

## 🐳 Run with Docker

```bash
docker build -t vehicle-cnn .
docker run -p 8501:8501 vehicle-cnn
```

## 🧾 License
MIT License
