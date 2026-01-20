# 🩺 Health Monitoring System with Real-Time Alerts 🚨 

An AI-powered health monitoring system built during B.Tech Final Year Project. The system uses machine learning for predictive health analysis, real-time alerts via SMS/calls, and OCR-based step tracking. Features user authentication, fall detection, and health parameter monitoring with an interactive Streamlit web interface.

---

## 📌 Features

- **Health Parameter Prediction**: Based on HRV, Blood Pressure, Sugar, and SpO2.
- **Fall Detection**: Analyzes step count and stability to predict fall risk.
- **Real-Time Alerts**: Sends alerts via SMS if critical values are detected.
- **OCR Integration**: Extracts step count from fitness app screenshots using Tesseract.
- **User Authentication**: Patient login/registration system for personalized tracking.

---

## 🧠 Technologies Used

- **Python** | **Streamlit** | **scikit-learn**
- **OCR (Tesseract)** | **Twilio SMS API**
- **Matplotlib, Seaborn, Altair** (for visualization)
- **Machine Learning Models**: SVM, Random Forest, etc.

---

## 🗃️ Project Structure

Health-Monitoring-System-with-Real-Time-Alerts/ ├── app.py ├── ml.ipynb ├── requirements.txt ├── README.md
├── models/ │ └── svm_classifier.pkl ├── data/ │ ├── Healthdataset.csv │ ├── step_data.csv │ ├── patient_data.csv │ └── fall_detection_data.csv

---

## 🚀 Quick Start

### Prerequisites
- Python 3.8+
- pip (Python package manager)
- (Optional) Tesseract OCR for step counting from images

### Installation & Setup

1. **Clone the repository**
   ```bash
   git clone https://github.com/yourusername/Health-Monitoring-System.git
   cd Health-Monitoring-System
   ```

2. **Create virtual environment** (optional but recommended)
   ```bash
   python -m venv venv
   
   # On Windows
   venv\Scripts\activate
   
   # On Linux/Mac
   source venv/bin/activate
   ```

3. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```

4. **Configure credentials (Optional)**
   - Edit `.env` file to add:
     - Twilio credentials (for SMS/Call alerts)
     - Google Gemini API key (for Chatbot)
     - Tesseract OCR path (for Step counting)
   - Leave blank to use demo mode without alerts

5. **Run the application**
   ```bash
   streamlit run app.py
   ```
   - Access at: `http://localhost:8501`

### Usage

1. **Register/Login** - Create patient account with health details
2. **Health Monitor** - Enter vital parameters (BP, HRV, Sugar, SpO2)
3. **Fall Detection** - Monitor fall risk over multiple days
4. **Walking Stability** - Upload fitness app screenshots for step analysis
5. **Guidelines** - View health tips and app workflow
6. **Chatbot** - Get AI-powered health recommendations (requires Gemini API)
7. **ML Accuracy** - View model performance metrics

---

## 🔧 Configuration Guide

### Optional Features (Leave blank to disable)

| Feature | Setup | Status |
|---------|-------|--------|
| **SMS/Call Alerts** | Add Twilio credentials to `.env` | ⚠️ Optional |
| **Chatbot** | Add Gemini API key to `.env` | ⚠️ Optional |
| **Step Counting OCR** | Install Tesseract, set path in `.env` | ⚠️ Optional |

### Install Tesseract (Optional)
- **Windows**: Download from [UB-Mannheim/tesseract](https://github.com/UB-Mannheim/tesseract/wiki)
- **Linux**: `sudo apt-get install tesseract-ocr`
- **Mac**: `brew install tesseract`

---

## 📁 Project Structure

```
Health-Monitoring-System/
├── app.py                          # Main Streamlit application
├── ml.ipynb                        # ML model training notebook
├── requirements.txt                # Python dependencies
├── .env                            # Configuration with dummy data
├── .gitignore                      # Git ignore rules
├── README.md                       # This file
├── models/
│   └── svm_classifier.pkl          # Trained SVM model
├── data/
│   ├── Healthdataset.csv           # Training dataset
│   ├── step_data.csv               # Step tracking data
│   ├── patient_data.csv            # Patient profiles
│   └── fall_detection_data.csv     # Fall detection logs
├── Watch.png                       # App screenshot
├── Final Year Project PPT.pptx     # Presentation
└── Manzar.pdf                      # Documentation
```

---

## ✨ Project Status

✅ **Completed** - B.Tech Final Year Project  
🧪 **Fully Tested** - Production-ready  
📊 **Demonstrated** - During Final Year Examination  
🚀 **Live Ready** - Deploy anywhere with Python 3.8+  

---

## 🎯 Key Features

- ✅ Real-time health parameter monitoring
- ✅ AI-powered fall detection algorithm
- ✅ Optional SMS/Call alerts via Twilio
- ✅ OCR-based fitness tracking
- ✅ User authentication & profiles
- ✅ ML model accuracy display
- ✅ Interactive charts & visualizations
- ✅ AI chatbot for health advice
- ✅ Demo mode (works without API keys)

---

## 🛠️ Tech Stack

| Category | Technologies |
|----------|---------------|
| **Frontend** | Streamlit |
| **Backend** | Python 3.8+ |
| **ML/AI** | scikit-learn, TensorFlow |
| **APIs** | Twilio, Google Generative AI |
| **Data Processing** | Pandas, NumPy |
| **Visualization** | Matplotlib, Seaborn, Altair |
| **OCR** | Tesseract |

---

## 📝 License

This project was developed as part of B.Tech Final Year Project.  
Free to use for educational and personal purposes.

---

## 👨‍💻 Author

**Md Manzar Nizam**  
🔗 LinkedIn: https://www.linkedin.com/in/md-manzar-nizam
💻 GitHub: https://github.com/manzar79md
📧 Email: manzarnizammd@gmail.com
