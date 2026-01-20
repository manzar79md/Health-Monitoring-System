import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import altair as alt
import time
import pyttsx3
import os
from twilio.rest import Client
import google.generativeai as genai
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
import pickle
from PIL import Image
import pytesseract  # For OCR to extract text from images
import re
from dotenv import load_dotenv

from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = BASE_DIR / "data"
MODEL_DIR = BASE_DIR / "models"

data_path = DATA_DIR / "Healthdataset.csv"
model_path = MODEL_DIR / "svm_classifier.pkl"

step_data_file = DATA_DIR / "step_data.csv"
patient_data_file = DATA_DIR / "patient_data.csv"
fall_data_file = DATA_DIR / "fall_detection_data.csv"

# Load environment variables from .env file
load_dotenv()

# Configure Tesseract OCR path from environment variable
tesseract_cmd = os.getenv("TESSERACT_CMD")
if tesseract_cmd:
    pytesseract.pytesseract.tesseract_cmd = tesseract_cmd

# Set page config
st.set_page_config(
    page_title="Health Monitoring System with Real Time Alerts",
    page_icon="❤️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Apply custom CSS to make it look more like the React UI
st.markdown("""
<style>
    /* Main Styles */
    .main .block-container {
        padding-top: 2rem;
        padding-bottom: 2rem;
    }
    
    /* Header Styles */
    .header {
        background-color: rgba(255, 255, 255, 0.8);
        backdrop-filter: blur(10px);
        padding: 1rem;
        border-bottom: 1px solid rgba(0, 0, 0, 0.1);
        margin-bottom: 2rem;
        display: flex;
        align-items: center;
        justify-content: space-between;
    }
    
    .app-title {
        font-size: 1.5rem;
        font-weight: 600;
        color: #1E88E5;
        display: flex;
        align-items: center;
        gap: 0.5rem;
    }
    
    /* Card Styles */
    .stCard {
        border-radius: 12px;
        box-shadow: 0 4px 6px rgba(0, 0, 0, 0.05), 0 1px 3px rgba(0, 0, 0, 0.1);
        padding: 1.5rem;
        background-color: white;
        margin-bottom: 1.5rem;
    }
    
    .card-title {
        font-size: 1.2rem;
        font-weight: 600;
        margin-bottom: 1rem;
        color: #333;
    }
    
    .card-description {
        color: #666;
        font-size: 0.9rem;
        margin-bottom: 1.5rem;
    }
    
    /* Metric Card Styles */
    .metric-card {
        padding: 1rem;
        border-radius: 0.75rem;
        text-align: center;
        height: 100%;
    }
    
    .metric-card-good {
        border: 1px solid rgba(52, 199, 89, 0.2);
        background-color: rgba(52, 199, 89, 0.05);
        color: #34C759;
    }
    
    .metric-card-warning {
        border: 1px solid rgba(255, 149, 0, 0.2);
        background-color: rgba(255, 149, 0, 0.05);
        color: #FF9500;
    }
    
    .metric-card-danger {
        border: 1px solid rgba(255, 59, 48, 0.2);
        background-color: rgba(255, 59, 48, 0.05);
        color: #FF3B30;
    }
    
    .metric-title {
        font-size: 1rem;
        font-weight: 500;
        display: flex;
        align-items: center;
        justify-content: center;
        gap: 0.5rem;
        margin-bottom: 0.5rem;
    }
    
    .metric-value {
        font-size: 2rem;
        font-weight: 700;
    }
    
    .metric-unit {
        font-size: 0.8rem;
        opacity: 0.8;
    }
    
    .metric-status {
        display: flex;
        align-items: center;
        justify-content: center;
        font-size: 0.8rem;
        gap: 0.25rem;
        margin-top: 0.5rem;
    }
    
    /* Form Styles */
    .form-field {
        margin-bottom: 1rem;
    }
    
    .form-label {
        font-weight: 500;
        display: flex;
        align-items: center;
        gap: 0.5rem;
        margin-bottom: 0.5rem;
    }
    
    .normal-range {
        font-size: 0.8rem;
        color: #666;
        margin-top: 0.25rem;
    }
    
    /* Button Styles */
    .stButton > button {
        background-color: #1E88E5;
        color: white;
        border-radius: 0.5rem;
        padding: 0.5rem 1rem;
        font-weight: 500;
        border: none;
        width: 100%;
        transition: all 0.3s;
    }
    
    .stButton > button:hover {
        background-color: #1976D2;
        box-shadow: 0 4px 8px rgba(0, 0, 0, 0.1);
    }
    
    /* Recommendation Styles */
    .recommendation-item {
        display: flex;
        align-items: flex-start;
        gap: 0.75rem;
        padding: 0.75rem;
        background-color: rgba(0, 0, 0, 0.03);
        border-radius: 0.5rem;
        margin-bottom: 0.5rem;
    }
    
    .recommendation-icon {
        flex-shrink: 0;
    }
    
    .recommendation-text {
        font-size: 0.9rem;
    }
    
    /* Apply gradient background like in the React app */
    .main {
        background: linear-gradient(180deg, #f8f9fa 0%, #e9ecef 100%);
    }
    
    /* Sidebar styles */
    .css-1d391kg {
        background-color: #f8f9fa;
    }
    
</style>
""", unsafe_allow_html=True)

# Header with logo
st.markdown("""
<div class="header">
    <div class="app-title">
        <svg width="24" height="24" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round">
            <polyline points="22 12 18 12 15 21 9 3 6 12 2 12"></polyline>
        </svg>
        Health Monitoring System with Real Time Alerts
    </div>
</div>
""", unsafe_allow_html=True)

# Twilio Credentials (Use environment variables for security)
TWILIO_ACCOUNT_SID = os.getenv("TWILIO_ACCOUNT_SID")
TWILIO_AUTH_TOKEN = os.getenv("TWILIO_AUTH_TOKEN")
TWILIO_PHONE_NUMBER = os.getenv("TWILIO_PHONE_NUMBER")
USER_PHONE_NUMBER = os.getenv("USER_PHONE_NUMBER")

# Check if Twilio credentials are available
TWILIO_ENABLED = all([TWILIO_ACCOUNT_SID, TWILIO_AUTH_TOKEN, TWILIO_PHONE_NUMBER, USER_PHONE_NUMBER])

# Initialize Twilio Client (only if credentials are available)
client = None
if TWILIO_ENABLED:
    try:
        client = Client(TWILIO_ACCOUNT_SID, TWILIO_AUTH_TOKEN)
    except Exception as e:
        st.warning(f"Twilio client initialization failed: {e}. SMS/Call alerts will be disabled.")
        TWILIO_ENABLED = False
else:
    st.warning("⚠️ Twilio credentials not configured. SMS and Call alerts are disabled. To enable, add credentials to your .env file.")

# Initialize Text-to-Speech engine
def speak(text):
    engine = pyttsx3.init()
    engine.say(text)
    engine.runAndWait()
    st.write(text)

# Function to detect fall based on health parameters
def detect_fall(row):
    try:
        bp = float(row["BP"]) if pd.notna(row["BP"]) else 0
        hrv = float(row["HRV"]) if pd.notna(row["HRV"]) else 0
        spo2 = float(row["SpO2"]) if pd.notna(row["SpO2"]) else 0

        if bp < 90 or hrv < 30 or spo2 < 85:
            return "Fall Detected"
        return "No Fall"
    except ValueError:
        return "Invalid Data"

# Function to send SMS alert
def send_sms(message):
    if not TWILIO_ENABLED:
        st.warning(f"📵 [SMS - Demo Mode] {message}")
        st.info("(SMS feature disabled - Twilio not configured)")
        return False
    try:
        client.messages.create(
            body=message,
            from_=TWILIO_PHONE_NUMBER,
            to=USER_PHONE_NUMBER
        )
        st.write(f"✅ [SMS Sent] {message}")
        return True
    except Exception as e:
        st.write(f"❌ Error sending SMS: {e}")
        return False

# Function to make call alert
def make_call():
    if not TWILIO_ENABLED:
        st.warning("☎️ [Call - Demo Mode] Call alert would be triggered")
        st.info("(Call feature disabled - Twilio not configured)")
        return False
    try:
        client.calls.create(
            twiml="<Response><Say>Alert! A fall has been detected. Immediate attention is required.</Say></Response>",
            from_=TWILIO_PHONE_NUMBER,
            to=USER_PHONE_NUMBER
        )
        st.write("✅ [Call Alert] Call initiated successfully.")
        return True
    except Exception as e:
        st.write(f"❌ Error making call: {e}")
        return False

# Gemini API key (for chatbot feature)
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")

# Check if Gemini API is available
GEMINI_ENABLED = False
if GEMINI_API_KEY:
    try:
        genai.configure(api_key=GEMINI_API_KEY)
        GEMINI_ENABLED = True
    except Exception as e:
        st.warning(f"Gemini API configuration failed: {e}. Chatbot will be disabled.")
        GEMINI_ENABLED = False
else:
    # Don't stop - just disable the chatbot feature
    GEMINI_ENABLED = False

# Function to generate chatbot responses
def chat_with_gemini(user_input):
    if not GEMINI_ENABLED:
        return "🤖 Chatbot feature is disabled. Please configure GEMINI_API_KEY in your .env file to enable the chatbot."
    try:
        # Initialize the model with a valid model name
        model = genai.GenerativeModel("gemini-1.5-flash")
        # Generate content with safety settings and generation config
        response = model.generate_content(
            user_input,
            generation_config={
                "temperature": 0.7,
                "max_output_tokens": 500
            },
            safety_settings={
                "HARM_CATEGORY_HARASSMENT": "BLOCK_MEDIUM_AND_ABOVE",
                "HARM_CATEGORY_HATE_SPEECH": "BLOCK_MEDIUM_AND_ABOVE",
                "HARM_CATEGORY_SEXUALLY_EXPLICIT": "BLOCK_MEDIUM_AND_ABOVE",
                "HARM_CATEGORY_DANGEROUS_CONTENT": "BLOCK_MEDIUM_AND_ABOVE"
            }
        )
        # Return the response text directly
        return response.text if response.text else "Sorry, I couldn't generate a response. Please try again."
    except Exception as e:
        return f"Error: {str(e)}"

# Load the dataset and model for ML accuracy
# data_path = "Healthdataset.csv"
# model_path = "svm_classifier.pkl"

def load_model():
    with open(model_path, "rb") as file:
        loaded_model = pickle.load(file)
    return loaded_model

def show_model_accuracy():
    data = pd.read_csv(data_path)
    data.columns = data.columns.str.strip()  # Clean column names
    X = data.drop(columns=['Decision'], errors='ignore')
    y = data['Decision']
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)
    model = load_model()
    y_pred = model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    st.subheader("📊 Model Accuracy")
    st.write(f"✅ **SVM Model Accuracy: {accuracy:.2%}**")  # Display as percentage

# Define the CSV file for storing user data
USER_DATA_FILE = patient_data_file

# Function to check if CSV file exists, if not create one
def initialize_user_data():
    if not os.path.exists(USER_DATA_FILE):
        df = pd.DataFrame(columns=["Username", "Password", "Phone Number", "Age", "Blood Group", "Food Allergies"])
        df.to_csv(USER_DATA_FILE, index=False)

# Default values for inputs
default_values = {"BP": 120.0, "HRV": 70.0, "Sugar Levels": 100.0, "SpO2": 95.0}
FALL_DETECTION_FILE = fall_data_file

# Initialize CSV file if not exists
def initialize_fall_data():
    if not os.path.exists(FALL_DETECTION_FILE):
        df = pd.DataFrame(columns=["Day", "BP", "HRV", "Sugar Levels", "SpO2", "Fall Status"])
        df.to_csv(FALL_DETECTION_FILE, index=False)

# Initialize session state for page navigation
if "page" not in st.session_state:
    st.session_state["page"] = "Home"

# Sidebar Navigation
with st.sidebar:
    st.title("Navigation")
    
    # Initialize session state for login
    if "logged_in" not in st.session_state:
        st.session_state["logged_in"] = False
    if "username" not in st.session_state:
        st.session_state["username"] = None

    initialize_user_data()  # Initialize user data
    initialize_fall_data()  # Initialize fall data

    # If not logged in, show login/register options
    if not st.session_state["logged_in"]:
        # Use session state to control the sidebar menu selection
        if "sidebar_menu" not in st.session_state:
            st.session_state["sidebar_menu"] = "Login"

        # Update sidebar menu based on page state
        if st.session_state["page"] == "Register":
            st.session_state["sidebar_menu"] = "Register"
        elif st.session_state["page"] == "Login":
            st.session_state["sidebar_menu"] = "Login"

        menu = st.radio("Menu", ["Login", "Register"], index=["Login", "Register"].index(st.session_state["sidebar_menu"]))

        if menu != st.session_state["sidebar_menu"]:
            st.session_state["sidebar_menu"] = menu
            st.session_state["page"] = menu
            st.rerun()

        if menu == "Login":
            st.subheader("Login")
            username = st.text_input("Username").strip()
            password = st.text_input("Password", type="password").strip()
            if st.button("Login"):
                if os.path.exists(USER_DATA_FILE):
                    user_data = pd.read_csv(USER_DATA_FILE, dtype={'Password': str})
                    user_row = user_data[user_data["Username"].str.strip() == username]

                    if not user_row.empty and user_row.iloc[0]["Password"].strip() == password:
                        st.session_state["logged_in"] = True
                        st.session_state["username"] = username
                        st.success(f"Logged in as {username}")
                        speak(f"Logged in as {username}")
                        st.session_state["page"] = "Home"  # Redirect to Home after login
                        st.rerun()
                    else:
                        st.error("Invalid username or password.")
                else:
                    st.error("No users registered. Please register first.")

        elif menu == "Register":
            st.subheader("Register New Patient")
            new_username = st.text_input("New Username")
            new_password = st.text_input("New Password", type="password")
            phone_number = st.text_input("Phone Number")
            age = st.number_input("Age", min_value=0, max_value=120, step=1)
            blood_group = st.selectbox("Blood Group", ["A+", "A-", "B+", "B-", "O+", "O-", "AB+", "AB-"])
            food_allergies = st.text_area("Food Allergies (comma-separated)")

            if st.button("Register"):
                if new_username and new_password and phone_number:
                    if os.path.exists(USER_DATA_FILE):
                        user_data = pd.read_csv(USER_DATA_FILE)
                        if new_username in user_data['Username'].values:
                            st.warning("Username already exists. Please choose a different username.")
                        else:
                            new_data = pd.DataFrame([[new_username, new_password, phone_number, age, blood_group, food_allergies]],
                                                    columns=["Username", "Password", "Phone Number", "Age", "Blood Group", "Food Allergies"])
                            user_data = pd.concat([user_data, new_data], ignore_index=True)
                            user_data.to_csv(USER_DATA_FILE, index=False)
                            st.success(f"User {new_username} registered successfully!")
                            speak(f"User {new_username} registered successfully!")
                            st.session_state["logged_in"] = True
                            st.session_state["username"] = new_username
                            st.session_state["page"] = "Home"  # Redirect to Home after registration
                            st.rerun()
                    else:
                        new_data = pd.DataFrame([[new_username, new_password, phone_number, age, blood_group, food_allergies]],
                                                columns=["Username", "Password", "Phone Number", "Age", "Blood Group", "Food Allergies"])
                        new_data.to_csv(USER_DATA_FILE, index=False)
                        st.success(f"User {new_username} registered successfully!")
                        speak(f"User {new_username} registered successfully!")
                        st.session_state["logged_in"] = True
                        st.session_state["username"] = new_username
                        st.session_state["page"] = "Home"  # Redirect to Home after registration
                        st.rerun()
                else:
                    st.warning("Please enter all details!")
    else:
        # Navigation links for logged-in users
        st.write(f"Logged in as: {st.session_state['username']}")
        selected = st.radio(
            "Go to",
            ["Home", "Health Monitor", "Fall Detection", "Walking Stability", "Guidelines", "Chatbot", "ML Accuracy"],
            index=0,
        )
        if selected != st.session_state["page"]:
            st.session_state["page"] = selected
            st.rerun()
        if st.button("Logout"):
            st.session_state["logged_in"] = False
            st.session_state["username"] = None
            st.session_state["page"] = "Home"
            st.session_state["sidebar_menu"] = "Login"
            st.rerun()

# Main content based on page state
if st.session_state["page"] == "Home":
    st.markdown("""
    <div style="text-align: center; margin-top: 2rem;">
        <h1 style="font-size: 2.5rem; font-weight: 700; color: #333;">YOUR PERSONAL HEALTH MONITORING ASSISTANT</h1>
        <p style="font-size: 1.1rem; color: #666; margin-bottom: 2rem;">
            Monitor your health parameters, track your well-being, and receive real-time alerts for potential health risks with our sophisticated monitoring system.
        </p>
    """, unsafe_allow_html=True)

    # Create columns for buttons to center them
    col1, col2, col3 = st.columns([1, 2, 1])
    with col2:
        if st.button("GET STARTED", key="get_started"):
            st.session_state["page"] = "Register"
            st.session_state["sidebar_menu"] = "Register"
            st.rerun()
        if st.button("LOGIN TO ACCOUNT", key="login"):
            st.session_state["page"] = "Login"
            st.session_state["sidebar_menu"] = "Login"
            st.rerun()

    st.markdown("""
    </div>
    """, unsafe_allow_html=True)

    # Feature Cards
    col1, col2, col3 = st.columns(3)
    with col1:
        st.markdown("""
        <div class="stCard">
            <div style="display: flex; align-items: center; gap: 0.5rem; margin-bottom: 1rem;">
                <svg width="24" height="24" viewBox="0 0 24 24" fill="none" stroke="#1E88E5" stroke-width="2" stroke-linecap="round" stroke-linejoin="round">
                    <path d="M20.84 4.61a5.5 5.5 0 0 0-7.78 0L12 5.67l-1.06-1.06a5.5 5.5 0 0 0-7.78 7.78l1.06 1.06L12 21.23l7.78-7.78 1.06-1.06a5.5 5.5 0 0 0 0-7.78z"></path>
                </svg>
                <div class="card-title">Health Monitoring</div>
            </div>
            <div class="card-description">
                Track important health parameters including blood pressure, heart rate, oxygen levels, and more in real-time.
            </div>
            <a href="#learn-more" style="color: #1E88E5; text-decoration: none; font-weight: 500;">Learn more </a>
        </div>
        """, unsafe_allow_html=True)

    with col2:
        st.markdown("""
        <div class="stCard">
            <div style="display: flex; align-items: center; gap: 0.5rem; margin-bottom: 1rem;">
                <svg width="24" height="24" viewBox="0 0 24 24" fill="none" stroke="#1E88E5" stroke-width="2" stroke-linecap="round" stroke-linejoin="round">
                    <path d="M12 2v4m0 12v4M2 12h4m12 0h4M4.93 4.93l2.83 2.83m8.48 8.48l2.83 2.83M4.93 19.07l2.83-2.83m8.48-8.48l2.83-2.83"></path>
                </svg>
                <div class="card-title">Fall Detection</div>
            </div>
            <div class="card-description">
                Advanced algorithm continuously analyzes your health data to predict and detect potential falls before they happen.
            </div>
            <a href="#learn-more" style="color: #1E88E5; text-decoration: none; font-weight: 500;">Learn more </a>
        </div>
        """, unsafe_allow_html=True)

    with col3:
        st.markdown("""
        <div class="stCard">
            <div style="display: flex; align-items: center; gap: 0.5rem; margin-bottom: 1rem;">
                <svg width="24" height="24" viewBox="0 0 24 24" fill="none" stroke="#1E88E5" stroke-width="2" stroke-linecap="round" stroke-linejoin="round">
                    <path d="M12 22s-8-4.5-8-11.8A8 8 0 0 1 12 2a8 8 0 0 1 8 8.2c0 7.3-8 11.8-8 11.8z"></path>
                    <circle cx="12" cy="10" r="3"></circle>
                </svg>
                <div class="card-title">Real-time Alerts</div>
            </div>
            <div class="card-description">
                Receive instant SMS and call alerts when critical health parameters are detected, ensuring timely intervention.
            </div>
            <a href="#learn-more" style="color: #1E88E5; text-decoration: none; font-weight: 500;">Learn more </a>
        </div>
        """, unsafe_allow_html=True)

    # CTA Section
    st.markdown("""
    <div style="background-color: #E3F2FD; padding: 2rem; border-radius: 1rem; text-align: center; margin-top: 2rem;">
        <h2 style="font-size: 1.75rem; font-weight: 600; color: #333;">Ready to start monitoring your health?</h2>
        <p style="color: #666; margin-bottom: 1.5rem;">
            Join thousands of users who trust our platform for their daily health monitoring and fall prevention needs.
        </p>
        <a href="#create-account" style="background-color: #1E88E5; color: white; padding: 0.75rem 1.5rem; border-radius: 0.5rem; text-decoration: none; font-weight: 500;">CREATE YOUR ACCOUNT</a>
    </div>
    """, unsafe_allow_html=True)

elif st.session_state["page"] == "Register" and not st.session_state["logged_in"]:
    st.markdown("""
    <div style="text-align: center; margin-top: 2rem;">
        <h2 style="font-size: 2rem; font-weight: 600; color: #333;">Register New Patient</h2>
        <p style="font-size: 1.1rem; color: #666; margin-bottom: 2rem;">
            Please use the sidebar to register a new patient account.
        </p>
    </div>
    """, unsafe_allow_html=True)

elif st.session_state["page"] == "Login" and not st.session_state["logged_in"]:
    st.markdown("""
    <div style="text-align: center; margin-top: 2rem;">
        <h2 style="font-size: 2rem; font-weight: 600; color: #333;">Login to Your Account</h2>
        <p style="font-size: 1.1rem; color: #666; margin-bottom: 2rem;">
            Please use the sidebar to log in to your account.
        </p>
    </div>
    """, unsafe_allow_html=True)

elif st.session_state["page"] == "Health Monitor" and st.session_state["logged_in"]:
    st.markdown('<div class="stCard">', unsafe_allow_html=True)
    st.markdown('<div class="card-title">Health Condition Monitoring</div>', unsafe_allow_html=True)
    st.markdown('<div class="card-description">Monitor your vital health parameters</div>', unsafe_allow_html=True)
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown('<div class="form-field">', unsafe_allow_html=True)
        st.markdown('''
        <div class="form-label">
            <svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="#0A84FF" stroke-width="2" stroke-linecap="round" stroke-linejoin="round">
                <polyline points="22 12 18 12 15 21 9 3 6 12 2 12"></polyline>
            </svg>
            Blood Pressure (BP)
        </div>
        ''', unsafe_allow_html=True)
        bp = st.number_input("", min_value=50, max_value=200, value=120, label_visibility="collapsed")
        st.markdown('<div class="normal-range">Normal range: 90-140 mmHg</div>', unsafe_allow_html=True)
        st.markdown('</div>', unsafe_allow_html=True)
        
        st.markdown('<div class="form-field">', unsafe_allow_html=True)
        st.markdown('''
        <div class="form-label">
            <svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="#0A84FF" stroke-width="2" stroke-linecap="round" stroke-linejoin="round">
                <path d="M20.84 4.61a5.5 5.5 0 0 0-7.78 0L12 5.67l-1.06-1.06a5.5 5.5 0 0 0-7.78 7.78l1.06 1.06L12 21.23l7.78-7.78 1.06-1.06a5.5 5.5 0 0 0 0-7.78z"></path>
            </svg>
            Heart Rate Variability (HRV)
        </div>
        ''', unsafe_allow_html=True)
        hrv = st.number_input("", min_value=0, max_value=100, value=50, label_visibility="collapsed")
        st.markdown('<div class="normal-range">Normal range: >20 ms</div>', unsafe_allow_html=True)
        st.markdown('</div>', unsafe_allow_html=True)
    
    with col2:
        st.markdown('<div class="form-field">', unsafe_allow_html=True)
        st.markdown('''
        <div class="form-label">
            <svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="#0A84FF" stroke-width="2" stroke-linecap="round" stroke-linejoin="round">
                <path d="M12 2.69l5.66 5.66a8 8 0 1 1-11.31 0z"></path>
            </svg>
            Sugar Levels
        </div>
        ''', unsafe_allow_html=True)
        sugar_level = st.number_input("", min_value=0, max_value=300, value=100, label_visibility="collapsed")
        st.markdown('<div class="normal-range">Normal range: 70-180 mg/dL</div>', unsafe_allow_html=True)
        st.markdown('</div>', unsafe_allow_html=True)
        
        st.markdown('<div class="form-field">', unsafe_allow_html=True)
        st.markdown('''
        <div class="form-label">
            <svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="#0A84FF" stroke-width="2" stroke-linecap="round" stroke-linejoin="round">
                <path d="M9.59 4.59A2 2 0 1 1 11 8H2m10.59 11.41A2 2 0 1 0 14 16H2m15.73-8.27A2.5 2.5 0 1 1 19.5 12H2"></path>
            </svg>
            SpO2 Level
        </div>
        ''', unsafe_allow_html=True)
        spo2 = st.number_input("", min_value=50, max_value=100, value=95, label_visibility="collapsed")
        st.markdown('<div class="normal-range">Normal range: >90%</div>', unsafe_allow_html=True)
        st.markdown('</div>', unsafe_allow_html=True)
    
    check_health = st.button("Check Health Condition")
    st.markdown('</div>', unsafe_allow_html=True)
    
    if check_health:
        progress_bar = st.progress(0)
        for i in range(100):
            time.sleep(0.01)
            progress_bar.progress(i + 1)
        
        messages = []
        speech_messages = []
        
        if bp < 90:
            bp_status = "danger"
            messages.append("Low blood pressure detected. Drink water, eat something salty, and rest.")
            speech_messages.append("Warning! Low blood pressure detected. Drink water, eat something salty, and rest. If symptoms persist, consult a doctor.")
        elif bp > 140:
            bp_status = "warning"
            messages.append("High blood pressure detected. Take prescribed medication and avoid salty foods.")
            speech_messages.append("Warning! High blood pressure detected. Take your prescribed medication, avoid salty foods, and relax. Consult a doctor if needed.")
        else:
            bp_status = "good"
            messages.append("Blood pressure is in normal range.")
            speech_messages.append("Blood pressure is in normal range.")
        
        if hrv < 20:
            hrv_status = "danger"
            messages.append("Low heart rate variability detected. Consider relaxation techniques.")
            speech_messages.append("Warning! Low heart rate variability detected. Consider relaxation techniques and consult a doctor if needed.")
        else:
            hrv_status = "good"
            messages.append("HRV levels are normal.")
            speech_messages.append("HRV levels are normal.")
        
        if sugar_level < 70:
            sugar_status = "danger"
            messages.append("Low blood sugar detected. Take glucose or eat something sugary immediately.")
            speech_messages.append("Warning! Low blood sugar detected. Take glucose or eat something sugary immediately.")
        elif sugar_level > 180:
            sugar_status = "warning"
            messages.append("High blood sugar detected. Take prescribed medication and follow a balanced diet.")
            speech_messages.append("Warning! High blood sugar detected. Take your prescribed medication and follow a balanced diet.")
        else:
            sugar_status = "good"
            messages.append("Blood sugar is in normal range.")
            speech_messages.append("Blood sugar is in normal range.")
        
        if spo2 < 90:
            spo2_status = "danger"
            messages.append("Low oxygen levels detected. Take deep breaths and consider oxygen support if needed.")
            speech_messages.append("Warning! Low oxygen levels detected. Take deep breaths, rest, and consider using oxygen support if needed.")
        else:
            spo2_status = "good"
            messages.append("Oxygen levels are normal.")
            speech_messages.append("Oxygen levels are normal.")
        
        if "danger" in [bp_status, hrv_status, sugar_status, spo2_status]:
            st.error("Critical health conditions detected. Please take necessary action.")
            speak("Critical health conditions detected. Please take necessary action immediately.")
            st.write("Triggering SMS and Call for critical condition...")
            sms_success = send_sms("ALERT: Critical health conditions detected! Immediate attention required.")
            call_success = make_call()
            if not sms_success or not call_success:
                st.warning("Failed to send SMS or make call. Please check Twilio configuration.")
        
        speak(" ".join(speech_messages))
        
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            bp_icon_path = '<path d="M22 11.08V12a10 10 0 1 1-5.93-9.14"></path><polyline points="22 4 12 14.01 9 11.01"></polyline>' if bp_status == "good" else '<circle cx="12" cy="12" r="10"></circle><line x1="12" y1="8" x2="12" y2="12"></line><line x1="12" y1="16" x2="12.01" y2="16"></line>'
            bp_status_text = "Normal range" if bp_status == "good" else "Elevated" if bp_status == "warning" else "Critical"
            
            st.markdown(f'''
            <div class="metric-card metric-card-{bp_status}">
                <div class="metric-title">
                    <svg width="20" height="20" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round">
                        <polyline points="22 12 18 12 15 21 9 3 6 12 2 12"></polyline>
                    </svg>
                    Blood Pressure
                </div>
                <div class="metric-value">{bp}</div>
                <div class="metric-unit">mmHg</div>
                <div class="metric-status">
                    <svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round">
                        {bp_icon_path}
                    </svg>
                    {bp_status_text}
                </div>
            </div>
            ''', unsafe_allow_html=True)
        
        with col2:
            hrv_icon_path = '<path d="M22 11.08V12a10 10 0 1 1-5.93-9.14"></path><polyline points="22 4 12 14.01 9 11.01"></polyline>' if hrv_status == "good" else '<circle cx="12" cy="12" r="10"></circle><line x1="12" y1="8" x2="12" y2="12"></line><line x1="12" y1="16" x2="12.01" y2="16"></line>'
            hrv_status_text = "Normal range" if hrv_status == "good" else "Moderate risk" if hrv_status == "warning" else "High risk"
            
            st.markdown(f'''
            <div class="metric-card metric-card-{hrv_status}">
                <div class="metric-title">
                    <svg width="20" height="20" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round">
                        <path d="M20.84 4.61a5.5 5.5 0 0 0-7.78 0L12 5.67l-1.06-1.06a5.5 5.5 0 0 0-7.78 7.78l1.06 1.06L12 21.23l7.78-7.78 1.06-1.06a5.5 5.5 0 0 0 0-7.78z"></path>
                    </svg>
                    HRV
                </div>
                <div class="metric-value">{hrv}</div>
                <div class="metric-unit">ms</div>
                <div class="metric-status">
                    <svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round">
                        {hrv_icon_path}
                    </svg>
                    {hrv_status_text}
                </div>
            </div>
            ''', unsafe_allow_html=True)
        
        with col3:
            sugar_icon_path = '<path d="M22 11.08V12a10 10 0 1 1-5.93-9.14"></path><polyline points="22 4 12 14.01 9 11.01"></polyline>' if sugar_status == "good" else '<circle cx="12" cy="12" r="10"></circle><line x1="12" y1="8" x2="12" y2="12"></line><line x1="12" y1="16" x2="12.01" y2="16"></line>'
            sugar_status_text = "Normal range" if sugar_status == "good" else "Moderate risk" if sugar_status == "warning" else "High risk"
            
            st.markdown(f'''
            <div class="metric-card metric-card-{sugar_status}">
                <div class="metric-title">
                    <svg width="20" height="20" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round">
                        <path d="M12 2.69l5.66 5.66a8 8 0 1 1-11.31 0z"></path>
                    </svg>
                    Blood Sugar
                </div>
                <div class="metric-value">{sugar_level}</div>
                <div class="metric-unit">mg/dL</div>
                <div class="metric-status">
                    <svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round">
                        {sugar_icon_path}
                    </svg>
                    {sugar_status_text}
                </div>
            </div>
            ''', unsafe_allow_html=True)
        
        with col4:
            spo2_icon_path = '<path d="M22 11.08V12a10 10 0 1 1-5.93-9.14"></path><polyline points="22 4 12 14.01 9 11.01"></polyline>' if spo2_status == "good" else '<circle cx="12" cy="12" r="10"></circle><line x1="12" y1="8" x2="12" y2="12"></line><line x1="12" y1="16" x2="12.01" y2="16"></line>'
            spo2_status_text = "Normal range" if spo2_status == "good" else "Moderate risk" if spo2_status == "warning" else "High risk"
            
            st.markdown(f'''
            <div class="metric-card metric-card-{spo2_status}">
                <div class="metric-title">
                    <svg width="20" height="20" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round">
                        <path d="M9.59 4.59A2 2 0 1 1 11 8H2m10.59 11.41A2 2 0 1 0 14 16H2m15.73-8.27A2.5 2.5 0 1 1 19.5 12H2"></path>
                    </svg>
                    SpO2
                </div>
                <div class="metric-value">{spo2}</div>
                <div class="metric-unit">%</div>
                <div class="metric-status">
                    <svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round">
                        {spo2_icon_path}
                    </svg>
                    {spo2_status_text}
                </div>
            </div>
            ''', unsafe_allow_html=True)
        
        st.markdown('<div class="stCard">', unsafe_allow_html=True)
        st.markdown('<div class="card-title">Health Recommendations</div>', unsafe_allow_html=True)
        
        for message in messages:
            if "normal" in message:
                icon_color = "#34C759"
                icon_path = '<path d="M22 11.08V12a10 10 0 1 1-5.93-9.14"></path><polyline points="22 4 12 14.01 9 11.01"></polyline>'
            else:
                icon_color = "#FF3B30"
                icon_path = '<circle cx="12" cy="12" r="10"></circle><line x1="12" y1="8" x2="12" y2="12"></line><line x1="12" y1="16" x2="12.01" y2="16"></line>'
            
            st.markdown(f'''
            <div class="recommendation-item">
                <svg class="recommendation-icon" width="20" height="20" viewBox="0 0 24 24" fill="none" stroke="{icon_color}" stroke-width="2" stroke-linecap="round" stroke-linejoin="round">
                    {icon_path}
                </svg>
                <div class="recommendation-text">{message}</div>
            </div>
            ''', unsafe_allow_html=True)
        
        st.markdown('</div>', unsafe_allow_html=True)
        
        st.markdown('<div class="stCard">', unsafe_allow_html=True)
        st.markdown('<div class="card-title">Health Parameters Visualization</div>', unsafe_allow_html=True)
        
        chart_data = pd.DataFrame({
            'Parameter': ['Blood Pressure', 'HRV', 'Blood Sugar', 'SpO2'],
            'Value': [bp, hrv, sugar_level, spo2],
            'Min': [90, 20, 70, 90],
            'Max': [140, 100, 180, 100],
            'Status': [bp_status, hrv_status, sugar_status, spo2_status]
        })
        
        chart_data['Normalized'] = 0
        chart_data.loc[0, 'Normalized'] = (bp - 50) / 150
        chart_data.loc[1, 'Normalized'] = hrv / 100
        chart_data.loc[2, 'Normalized'] = sugar_level / 300
        chart_data.loc[3, 'Normalized'] = (spo2 - 50) / 50
        
        chart_data['Normalized'] = chart_data['Normalized'].clip(0, 1)
        
        for i, row in chart_data.iterrows():
            param = row['Parameter']
            value = row['Value']
            min_val = row['Min']
            max_val = row['Max']
            status = row['Status']
            
            if status == 'good':
                color = '#34C759'
            elif status == 'warning':
                color = '#FF9500'
            else:
                color = '#FF3B30'
            
            fig, ax = plt.subplots(figsize=(3, 1.5))
            ax.barh(param, 1, color='#f0f0f0', alpha=0.3)
            ax.barh(param, row['Normalized'], color=color)
            ax.set_xlim(0, 1)
            ax.set_xlabel(f'{value} {["mmHg", "ms", "mg/dL", "%"][i]}')
            ax.set_title(param)
            ax.grid(False)
            
            fig.tight_layout()
            st.pyplot(fig)
        
        st.markdown('</div>', unsafe_allow_html=True)

elif st.session_state["page"] == "Fall Detection" and st.session_state["logged_in"]:
    st.markdown('<div class="stCard">', unsafe_allow_html=True)
    st.markdown('<div class="card-title">Fall Detection</div>', unsafe_allow_html=True)
    st.markdown('<div class="card-description">Monitor and detect potential falls in real-time</div>', unsafe_allow_html=True)
    
    st.subheader("Enter Health Parameters")
    
    days_to_monitor = st.slider("Number of days to monitor", 1, 15, 3)
    
    fall_data = []
    
    for i in range(days_to_monitor):
        st.markdown(f"### Day {i + 1}")
        col1, col2 = st.columns(2)
        
        with col1:
            bp = st.number_input(f"Blood Pressure (Day {i + 1})", min_value=50, max_value=200, value=120, key=f"bp_{i}")
            hrv = st.number_input(f"Heart Rate Variability (Day {i + 1})", min_value=0, max_value=100, value=70, key=f"hrv_{i}")
        
        with col2:
            sugar = st.number_input(f"Sugar Levels (Day {i + 1})", min_value=0, max_value=300, value=100, key=f"sugar_{i}")
            spo2 = st.number_input(f"SpO2 Level (Day {i + 1})", min_value=50, max_value=100, value=95, key=f"spo2_{i}")
        
        fall_data.append({
            "day": f"Day {i + 1}",
            "bp": bp,
            "hrv": hrv,
            "sugar": sugar,
            "spo2": spo2,
            "fallStatus": None
        })
    
    if st.button("Check Fall Prediction"):
        progress_bar = st.progress(0)
        for i in range(100):
            time.sleep(0.01)
            progress_bar.progress(i + 1)
        
        falls_detected = []
        
        for i, day_data in enumerate(fall_data):
            if day_data["bp"] < 90 or day_data["spo2"] < 90:
                fall_data[i]["fallStatus"] = "Fall Detected"
                falls_detected.append(day_data["day"])
            else:
                fall_data[i]["fallStatus"] = "No Fall"
        
        st.subheader("Fall Detection Results")
        
        results_df = pd.DataFrame(fall_data)
        
        def highlight_fall(s):
            return ['background-color: #ffcccc' if x == "Fall Detected" else 'background-color: #ccffcc' for x in s]
        
        styled_results = results_df.style.apply(highlight_fall, subset=['fallStatus'])
        st.dataframe(styled_results)
        
        if os.path.exists(FALL_DETECTION_FILE):
            fall_data_df = pd.read_csv(FALL_DETECTION_FILE)
        else:
            fall_data_df = pd.DataFrame(columns=["Day", "BP", "HRV", "Sugar Levels", "SpO2", "Fall Status"])
        
        new_data = pd.DataFrame({
            "Day": [d["day"] for d in fall_data],
            "BP": [d["bp"] for d in fall_data],
            "HRV": [d["hrv"] for d in fall_data],
            "Sugar Levels": [d["sugar"] for d in fall_data],
            "SpO2": [d["spo2"] for d in fall_data],
            "Fall Status": [d["fallStatus"] for d in fall_data]
        })
        fall_data_df = pd.concat([fall_data_df, new_data], ignore_index=True)
        fall_data_df.to_csv(FALL_DETECTION_FILE, index=False)
        st.success("Fall detection data saved successfully!")
        
        if falls_detected:
            st.error(f"Potential fall detected on {', '.join(falls_detected)}. Immediate attention required.")
            st.write("Triggering SMS and Call for fall detection...")
            sms_success = send_sms("ALERT: Fall detected! Immediate attention required.")
            call_success = make_call()
            if not sms_success or not call_success:
                st.warning("Failed to send SMS or make call. Please check Twilio configuration.")
            speak("Fall Detected! Immediate attention required.")
            
            st.markdown('<div class="stCard" style="border-color: #FF3B30; background-color: rgba(255, 59, 48, 0.05);">', unsafe_allow_html=True)
            st.markdown('<div class="card-title" style="color: #FF3B30;">Fall Alert</div>', unsafe_allow_html=True)
            
            st.markdown("""
            <p style="color: rgba(255, 59, 48, 0.9);">
                Potential fall detected based on your health parameters. Immediate attention may be required.
            </p>
            <div style="background-color: rgba(255, 59, 48, 0.1); padding: 1rem; border-radius: 0.5rem; margin-top: 1rem;">
                <p style="font-weight: 500; margin-bottom: 0.5rem;">Recommendations:</p>
                <ul style="margin-top: 0.5rem; list-style-type: none; padding-left: 0.5rem;">
                    <li style="display: flex; align-items: center; gap: 0.5rem; margin-bottom: 0.5rem;">
                        <span style="display: inline-block; width: 0.5rem; height: 0.5rem; border-radius: 50%; background-color: #FF3B30;"></span>
                        If you've fallen, call for help immediately
                    </li>
                    <li style="display: flex; align-items: center; gap: 0.5rem; margin-bottom: 0.5rem;">
                        <span style="display: inline-block; width: 0.5rem; height: 0.5rem; border-radius: 50%; background-color: #FF3B30;"></span>
                        Avoid sudden movements
                    </li>
                    <li style="display: flex; align-items: center; gap: 0.5rem; margin-bottom: 0.5rem;">
                        <span style="display: inline-block; width: 0.5rem; height: 0.5rem; border-radius: 50%; background-color: #FF3B30;"></span>
                        Check for injuries and seek medical attention
                    </li>
                    <li style="display: flex; align-items: center; gap: 0.5rem;">
                        <span style="display: inline-block; width: 0.5rem; height: 0.5rem; border-radius: 50%; background-color: #FF3B30;"></span>
                        Drink water and rest in a comfortable position
                    </li>
                </ul>
            </div>
            """, unsafe_allow_html=True)
            
            st.markdown('</div>', unsafe_allow_html=True)
        else:
            st.success("No falls detected. All health parameters indicate stable conditions.")
            speak("No fall detected. Stay safe!")
    
    st.markdown('</div>', unsafe_allow_html=True)

elif st.session_state["page"] == "Walking Stability" and st.session_state["logged_in"]:
    st.markdown('<div class="stCard">', unsafe_allow_html=True)
    st.markdown('<div class="card-title">Walking Stability</div>', unsafe_allow_html=True)
    st.markdown('<div class="card-description">Analyze walking patterns for improved stability and fall prevention</div>', unsafe_allow_html=True)

    # Load the dataset from CSV
    csv_file = step_data_file
    try:
        df = pd.read_csv(csv_file)
        df["Date"] = pd.to_datetime(df["Date"])
    except FileNotFoundError:
        st.error("step_data.csv not found. Please create the file with the initial dataset.")
        df = pd.DataFrame(columns=["Date", "Step Count", "Active Hours", "Steps Per Minute"])
        df["Date"] = pd.to_datetime(df["Date"])
    except Exception as e:
        st.error(f"Error loading dataset: {e}")
        df = pd.DataFrame(columns=["Date", "Step Count", "Active Hours", "Steps Per Minute"])
        df["Date"] = pd.to_datetime(df["Date"])

    # Initialize session state for uploaded image
    if "uploaded_image" not in st.session_state:
        st.session_state["uploaded_image"] = None

    # Image upload for step count data
    st.subheader("Upload Step Count Image")
    uploaded_image = st.file_uploader("Upload an image from Samsung Health (or similar app)", type=["png", "jpg", "jpeg"], key="image_uploader")
    
    # Update session state with the uploaded image
    if uploaded_image is not None:
        st.session_state["uploaded_image"] = uploaded_image

    # Add a Clear button to reset the uploaded image
    if st.session_state["uploaded_image"] is not None:
        col1, col2 = st.columns([1, 1])
        with col1:
            st.write(f"Uploaded: {st.session_state['uploaded_image'].name}")
        with col2:
            if st.button("Clear"):
                st.session_state["uploaded_image"] = None
                st.rerun()

    # Process the image if one is uploaded
    if st.session_state["uploaded_image"] is not None:
        step_count = None  # Safe default

        try:
            # Load image
            uploaded_file = st.session_state["uploaded_image"]
            image = Image.open(uploaded_file)

            if image is None:
                st.error("Could not open image file. Please re-upload.")
            else:
                # Run OCR
                text = pytesseract.image_to_string(image)

                # Show extracted text (optional for debug)
                st.write("📝 OCR Raw Output:")
                st.code(text)

                # Try to extract step count
                numbers = re.findall(r'\b\d{4,6}\b', re.sub(r'[^\d]', '', text))
                for num in numbers:
                    value = int(num)
                    if 1000 <= value <= 100000:
                        step_count = value
                        break

            if step_count:
                st.success(f"Step count detected: {step_count} steps")

                # Extract date
                # Look for formats like "12/03", "03-12", "12 Mar", etc.
                date_match = re.search(r'(\d{1,2}[/\- ]\d{1,2})', text)
                if date_match:
                    date_str = date_match.group(1)
                    try:
                        date = pd.to_datetime(date_str, dayfirst=True, errors='raise')
                    except:
                        date = pd.Timestamp.now().date()
                        st.warning("Could not parse date. Using current date.")


                else:
                    date = pd.Timestamp.now().date()
                    st.warning("Date not found in image. Using current date.")

                # Defaults
                active_hours = 6
                steps_per_minute = 0.0

                # Append data
                new_row = pd.DataFrame({
                    "Date": [pd.to_datetime(date)],
                    "Step Count": [step_count],
                    "Active Hours": [active_hours],
                    "Steps Per Minute": [steps_per_minute]
                })
                df = pd.concat([df, new_row], ignore_index=True)
                df.to_csv(csv_file, index=False)
                st.success(f"Data added for {date} from image!")

                # Step variability
                if not df.empty and df["Step Count"].notna().any() and df["Step Count"].mean():
                    try:
                        cv = (df["Step Count"].std() / df["Step Count"].mean()) * 100
                        step_variability_score = 100 if cv < 20 else max(0, 100 - (cv - 20) * 5)
                    except Exception as e:
                        st.error(f"Error calculating variability score: {e}")
                        step_variability_score = 0
                else:
                    step_variability_score = 0

                # Stability index
                total_steps = step_count
                target_steps = 6000
                step_activity_score = min(100, (total_steps / target_steps) * 100)

                if 100 <= steps_per_minute <= 120:
                    step_frequency_score = 100
                elif steps_per_minute < 100:
                    step_frequency_score = steps_per_minute
                else:
                    step_frequency_score = max(0, 100 - (steps_per_minute - 120) * 2)

                activity_distribution_score = 100 if active_hours >= 6 else (active_hours / 6) * 100

                stability_score = (
                    0.3 * step_activity_score +
                    0.3 * step_variability_score +
                    0.2 * step_frequency_score +
                    0.2 * activity_distribution_score
                )
                stability_score = round(stability_score)

                if stability_score >= 81:
                    stability_level = "Very Good"
                    stability_risk = "Very low fall risk"
                    stability_color = "#34C759"
                elif stability_score >= 61:
                    stability_level = "Good"
                    stability_risk = "Low fall risk"
                    stability_color = "#34C759"
                elif stability_score >= 41:
                    stability_level = "OK"
                    stability_risk = "Moderate fall risk"
                    stability_color = "#FF9500"
                elif stability_score >= 21:
                    stability_level = "Low"
                    stability_risk = "Elevated fall risk"
                    stability_color = "#FF9500"
                else:
                    stability_level = "Very Low"
                    stability_risk = "High fall risk"
                    stability_color = "#FF3B30"

                # Display prediction
                st.markdown(
                    f"""
                    <div style="padding: 1rem; border-radius: 0.5rem; border: 1px solid {stability_color}; background-color: rgba({int(stability_color[1:3], 16)}, {int(stability_color[3:5], 16)}, {int(stability_color[5:7], 16)}, 0.1); margin-top: 1rem;">
                        <h4 style="color: {stability_color}; margin-bottom: 0.5rem;">Fall Risk Prediction</h4>
                        <p><strong>Stability Level:</strong> {stability_level}</p>
                        <p><strong>Stability Score:</strong> {stability_score}/100</p>
                        <p><strong>Fall Risk:</strong> {stability_risk}</p>
                    </div>
                    """,
                    unsafe_allow_html=True
                )
            else:
                st.error("Could not detect step count in the image. Please ensure it is clearly visible.")
        except Exception as e:
            st.error(f"An error occurred while processing the image: {e}")

            # Form to add new data manually
            st.subheader("Add New Step Count Data")
            with st.form(key="add_data_form"):
                col1_form, col2_form = st.columns(2)
            with col1_form:
                new_date = st.date_input("Date", value=pd.Timestamp("2025-03-18"))
                new_steps = st.number_input("Total Steps", min_value=0, max_value=20000, value=0)
            with col2_form:
                new_active_hours = st.number_input("Active Hours", min_value=0, max_value=24, value=6)
                new_spm = st.number_input("Steps per Minute (SPM)", min_value=0.0, max_value=200.0, value=0.0, step=0.1)
                submit_button = st.form_submit_button("Add Data")

            if submit_button:
                with st.spinner("Adding data..."):
                    try:
                    # Validate inputs
                        if new_steps < 0 or new_active_hours < 0 or new_spm < 0:
                            st.error("All values must be non-negative.")
                        else:
                        # Append new data to the DataFrame
                            new_row = pd.DataFrame({
                            "Date": [pd.to_datetime(new_date)],
                            "Step Count": [new_steps],
                            "Active Hours": [new_active_hours],
                            "Steps Per Minute": [new_spm]
                        })
                        df = pd.concat([df, new_row], ignore_index=True)
                        # Save the updated DataFrame back to the CSV
                        df.to_csv(csv_file, index=False)
                        st.success(f"Data added for {new_date}!")
                    except Exception as e:
                        st.error(f"Error adding data: {e}")

                    # Calculate Step Variability Score (based on the entire dataset) for later use
    if not df.empty and df["Step Count"].notna().any():
        try:
            cv = (df["Step Count"].std() / df["Step Count"].mean()) * 100
            step_variability_score = 100 if cv < 20 else max(0, 100 - (cv - 20) * 5)
        except Exception as e:
            st.error(f"Error calculating variability score: {e}")
            step_variability_score = 0
    else:
        step_variability_score = 0

    # Initialize stability_score in session state
    if "stability_score" not in st.session_state:
        st.session_state["stability_score"] = 75

    col1, col2 = st.columns(2)

    with col1:
        # Select a date for analysis
        st.subheader("Select a Date to Analyze Walking Steadiness")
        if not df.empty:
            # Convert dates to string format for selectbox
            date_options = df["Date"].dt.strftime("%Y-%m-%d").tolist()
            selected_date = st.selectbox("Date", date_options)
            # Fetch data for the selected date
            selected_data = df[df["Date"] == pd.to_datetime(selected_date)].iloc[0]
            total_steps = selected_data["Step Count"]
            active_hours = selected_data["Active Hours"]
            spm = selected_data["Steps Per Minute"]

            st.write(f"**Step Count Data for {selected_date}:**")
            st.write(f"Total Steps: {total_steps}")
            st.write(f"Active Hours: {active_hours}")
            st.write(f"Steps per Minute (SPM): {round(spm, 1)}")

            if st.button("Analyze Walking Steadiness"):
                with st.spinner("Analyzing walking steadiness..."):
                    try:
                        # Calculate Stability Index
                        target_steps = 6000
                        step_activity_score = min(100, (total_steps / target_steps) * 100)

                        # Step Frequency Score
                        if 100 <= spm <= 120:
                            step_frequency_score = 100
                        elif spm < 100:
                            step_frequency_score = spm
                        else:
                            step_frequency_score = max(0, 100 - (spm - 120) * 2)

                        # Activity Distribution Score
                        activity_distribution_score = 100 if active_hours >= 6 else (active_hours / 6) * 100

                        # Stability Index
                        stability_score = (
                            0.3 * step_activity_score +
                            0.3 * step_variability_score +
                            0.2 * step_frequency_score +
                            0.2 * activity_distribution_score
                        )
                        stability_score = round(stability_score)

                        # Update the stability score in session state
                        st.session_state["stability_score"] = stability_score

                        # Categorize stability and provide recommendations
                        if stability_score >= 81:
                            stability_level = "Very Good"
                            stability_risk = "Very low fall risk"
                            stability_color = "#34C759"
                            stability_recommendations = [
                                "Maintain current fitness level",
                                "Consider challenging balance activities",
                                "Help others with stability exercises",
                                "Continue regular physical activity"
                            ]
                        elif stability_score >= 61:
                            stability_level = "Good"
                            stability_risk = "Low fall risk"
                            stability_color = "#34C759"
                            stability_recommendations = [
                                "Maintain current exercise routine",
                                "Continue strength training",
                                "Stay active daily",
                                "Monitor any changes in stability"
                            ]
                        elif stability_score >= 41:
                            stability_level = "OK"
                            stability_risk = "Moderate fall risk"
                            stability_color = "#FF9500"
                            stability_recommendations = [
                                "Regular balance exercises",
                                "Stay physically active",
                                "Be cautious in new environments",
                                "Consider stability-focused workouts"
                            ]
                        elif stability_score >= 21:
                            stability_level = "Low"
                            stability_risk = "Elevated fall risk"
                            stability_color = "#FF9500"
                            stability_recommendations = [
                                "Take care on uneven surfaces",
                                "Consider balance exercises",
                                "Use handrails on stairs",
                                "Wear supportive footwear"
                            ]
                        else:
                            stability_level = "Very Low"
                            stability_risk = "High fall risk"
                            stability_color = "#FF3B30"
                            stability_recommendations = [
                                "Use a walking aid (cane, walker)",
                                "Avoid walking alone",
                                "Consider physical therapy",
                                "Ensure home environment is free of hazards"
                            ]

                        # Generate recommendations HTML separately
                        recommendations_html = "".join(
                            f'<li style="display: flex; align-items: center; gap: 0.5rem; margin-bottom: 0.5rem;">'
                            f'<span style="display: inline-block; width: 0.5rem; height: 0.5rem; border-radius: 50%; background-color: {stability_color};"></span>'
                            f'{rec}</li>'
                            for rec in stability_recommendations
                        )

                        # Display the analysis
                        st.markdown(
                            f"""
                            <div style="padding: 1.5rem; border-radius: 0.75rem; border: 1px solid {stability_color}; background-color: rgba({int(stability_color[1:3], 16)}, {int(stability_color[3:5], 16)}, {int(stability_color[5:7], 16)}, 0.1); margin-top: 1rem;">
                                <h3 style="display: flex; align-items: center; gap: 0.5rem; font-size: 1.25rem; margin-bottom: 1rem; color: {stability_color};">
                                    <svg width="24" height="24" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round">
                                        <path d="M13 4v16"></path><path d="M17 4v16"></path><path d="M21 4v16"></path><path d="M9 4v16"></path><path d="M5 4v16"></path><path d="M1 4v16"></path>
                                    </svg>
                                    Walking Stability: {stability_level}
                                </h3>
                                <div style="background-color: rgba({int(stability_color[1:3], 16)}, {int(stability_color[3:5], 16)}, {int(stability_color[5:7], 16)}, 0.1); padding: 1rem; border-radius: 0.5rem; margin-bottom: 1rem;">
                                    <div style="display: flex; align-items: center; gap: 0.5rem; font-weight: 500; margin-bottom: 0.5rem;">
                                        <svg width="20" height="20" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round">
                                            <polyline points="22 12 18 12 15 21 9 3 6 12 2 12"></polyline>
                                        </svg>
                                        Stability Score: {stability_score}/100
                                    </div>
                                    <p>{stability_risk}</p>
                                </div>
                                <h4 style="font-weight: 500; margin-bottom: 0.75rem;">Recommendations:</h4>
                                <ul style="list-style-type: none; padding-left: 0.25rem;">
                                    {recommendations_html}
                                </ul>
                                <p style="font-size: 0.75rem; color: #666; margin-top: 1rem;">
                                    This analysis is based on your step count data and is for informational purposes only. 
                                    Please consult a healthcare professional for a comprehensive assessment.
                                </p>
                            </div>
                            """,
                            unsafe_allow_html=True
                        )

                        # Create a progress chart showing stability over time
                        st.subheader("Stability Trend")
                        df["Stability Index"] = df.apply(
                            lambda row: round(
                                0.3 * min(100, (row["Step Count"] / 6000) * 100) +
                                0.3 * step_variability_score +
                                0.2 * (100 if 100 <= row["Steps Per Minute"] <= 120 else (row["Steps Per Minute"] if row["Steps Per Minute"] < 100 else max(0, 100 - (row["Steps Per Minute"] - 120) * 2))) +
                                0.2 * (100 if row["Active Hours"] >= 6 else (row["Active Hours"] / 6) * 100)
                            ), axis=1
                        )
                        chart = alt.Chart(df).mark_line(point=True).encode(
                            x='Date:T',
                            y=alt.Y('Stability Index:Q', scale=alt.Scale(domain=[0, 100])),
                            tooltip=['Date:T', 'Stability Index:Q']
                        ).properties(
                            title='Stability Trend',
                            width=600,
                            height=300
                        )
                        st.altair_chart(chart, use_container_width=True)
                    except Exception as e:
                        st.error(f"Error analyzing walking steadiness: {e}")
        else:
            st.warning("No data available. Please add data to analyze walking stability.")

    with col2:
        if st.button("Simulate Device Data"):
            # Generate a random stability score
            new_stability = np.random.randint(30, 95)
            
            # Display update message
            st.success(f"Device data updated! New stability score: {new_stability}")
            
            # Update the stability score in session state
            st.session_state["stability_score"] = new_stability
            st.rerun()

    # Display the stability meter
    st.subheader("Stability Meter")
    
    # Create the meter visualization
    fig, ax = plt.subplots(figsize=(8, 2))
    
    # Draw the background
    ax.barh(0, 100, color='#f0f0f0', alpha=0.3, height=0.3)
    
    # Draw colored sections
    ax.barh(0, 20, color='#FF3B30', alpha=0.7, height=0.3)
    ax.barh(0, 20, left=20, color='#FF9500', alpha=0.7, height=0.3)
    ax.barh(0, 20, left=40, color='#FFCC00', alpha=0.7, height=0.3)
    ax.barh(0, 20, left=60, color='#A2D729', alpha=0.7, height=0.3)
    ax.barh(0, 20, left=80, color='#34C759', alpha=0.7, height=0.3)
    
    # Draw labels
    ax.text(10, -0.25, 'Very Low', ha='center', va='top', fontsize=8)
    ax.text(30, -0.25, 'Low', ha='center', va='top', fontsize=8)
    ax.text(50, -0.25, 'OK', ha='center', va='top', fontsize=8)
    ax.text(70, -0.25, 'Good', ha='center', va='top', fontsize=8)
    ax.text(90, -0.25, 'Very Good', ha='center', va='top', fontsize=8)
    
    # Draw the indicator for current stability score
    ax.axvline(st.session_state["stability_score"], color='#1E88E5', linestyle='--', linewidth=2)
    ax.text(st.session_state["stability_score"], 0.4, f'Score: {st.session_state["stability_score"]}', 
            ha='center', va='bottom', fontsize=10, color='#1E88E5')
    
    # Customize the plot
    ax.set_xlim(0, 100)
    ax.set_ylim(-0.5, 0.5)
    ax.set_yticks([])
    ax.set_xlabel('Stability Score')
    ax.grid(False)
    
    # Display the plot
    st.pyplot(fig)
    
    st.markdown('</div>', unsafe_allow_html=True)
    
elif st.session_state["page"] == "Guidelines" and st.session_state["logged_in"]:
    st.markdown('<div class="stCard">', unsafe_allow_html=True)
    st.markdown('<div class="card-title">Health Guidelines</div>', unsafe_allow_html=True)
    st.markdown('<div class="card-description">Learn how to use this platform and improve your health</div>', unsafe_allow_html=True)
    
    # Virtual Doctor Explanation
    st.markdown("### 🏥 Virtual Doctor Explanation")
    st.write("The virtual doctor analyzes your health parameters and provides recommendations based on the readings.")
    speak("Welcome to the health guidelines section. The virtual doctor will analyze your health parameters and provide recommendations.")
    
    # Website Workflow Guide
    st.markdown("### 🔄 Website Workflow Guide")
    st.write("1. **Login or Register** to access your health monitoring system.")
    st.write("2. **Health Monitor**: Track your vital health parameters like BP, HRV, sugar levels, and SpO2.")
    st.write("3. **Fall Detection**: Monitor potential falls based on your health data and receive real-time alerts.")
    st.write("4. **Walking Stability**: Analyze your walking patterns to improve stability and prevent falls.")
    st.write("5. **Chatbot**: Ask health-related questions and get instant advice from our AI-powered assistant.")
    st.write("6. **ML Accuracy**: Evaluate the performance of the fall prediction model.")
    speak("This website allows you to monitor your health, check for falls, analyze walking stability, and receive AI-powered advice.")
    
    # Speech Output for Accessibility
    st.markdown("### 🗣️ Speech Output for Accessibility")
    st.write("For better accessibility, speech output is available for all major alerts, recommendations, and guidelines.")
    speak("Speech output is enabled to ensure accessibility for all users. Stay informed with real-time audio updates.")
    
    # General Health Tips
    st.markdown("### 💡 General Health Tips")
    st.markdown("""
    <div style="background-color: rgba(0, 0, 0, 0.03); padding: 1rem; border-radius: 0.5rem;">
        <ul style="list-style-type: none; padding-left: 0.5rem;">
            <li style="display: flex; align-items: center; gap: 0.5rem; margin-bottom: 0.5rem;">
                <span style="display: inline-block; width: 0.5rem; height: 0.5rem; border-radius: 50%; background-color: #1E88E5;"></span>
                Stay hydrated by drinking at least 8 glasses of water daily.
            </li>
            <li style="display: flex; align-items: center; gap: 0.5rem; margin-bottom: 0.5rem;">
                <span style="display: inline-block; width: 0.5rem; height: 0.5rem; border-radius: 50%; background-color: #1E88E5;"></span>
                Engage in at least 30 minutes of physical activity most days of the week.
            </li>
            <li style="display: flex; align-items: center; gap: 0.5rem; margin-bottom: 0.5rem;">
                <span style="display: inline-block; width: 0.5rem; height: 0.5rem; border-radius: 50%; background-color: #1E88E5;"></span>
                Monitor your health parameters regularly and consult a doctor if you notice any unusual changes.
            </li>
            <li style="display: flex; align-items: center; gap: 0.5rem;">
                <span style="display: inline-block; width: 0.5rem; height: 0.5rem; border-radius: 50%; background-color: #1E88E5;"></span>
                Ensure your home is free of hazards to prevent falls, such as removing clutter and securing rugs.
            </li>
        </ul>
    </div>
    """, unsafe_allow_html=True)
    
    st.markdown('</div>', unsafe_allow_html=True)


elif st.session_state["page"] == "Chatbot" and st.session_state["logged_in"]:
    st.markdown('<div class="stCard">', unsafe_allow_html=True)
    st.markdown('<div class="card-title">Chat with Health Assistant</div>', unsafe_allow_html=True)
    st.markdown('<div class="card-description">Ask health-related questions and get instant advice</div>', unsafe_allow_html=True)
    
    # Chatbot UI
    st.subheader("💬 AI Chatbot")
    
    # Initialize chat history in session state
    if "chat_history" not in st.session_state:
        st.session_state["chat_history"] = []

    # Display chat history
    for message in st.session_state["chat_history"]:
        if message["role"] == "user":
            st.markdown(f"""
            <div style="background-color: #E3F2FD; padding: 1rem; border-radius: 0.5rem; margin-bottom: 0.5rem; max-width: 70%; margin-left: auto;">
                <strong>You:</strong> {message["content"]}
            </div>
            """, unsafe_allow_html=True)
        else:
            # Process the assistant's response to handle markdown-like formatting
            # First replace newlines, then handle bold formatting, then bullets
            formatted_response = message["content"]
            formatted_response = formatted_response.replace('\n\n', '<br><br>')  # Paragraph breaks
            formatted_response = formatted_response.replace('**', '<strong>', 1)  # First ** to <strong>
            formatted_response = formatted_response.replace('**', '</strong>', 1)  # Second ** to </strong>
            formatted_response = formatted_response.replace('*', '• ')  # Bullets
            st.markdown(f"""
            <div style="background-color: #F5F5F5; padding: 1rem; border-radius: 0.5rem; margin-bottom: 0.5rem; max-width: 70%;">
                <strong>Health Assistant:</strong> {formatted_response}
            </div>
            """, unsafe_allow_html=True)
    
    # User input field
    user_input = st.text_input("Ask a health-related question:", placeholder="e.g., What should I do if my blood pressure is high?", key="chat_input")

    if st.button("Send"):
        if user_input:
            # Add user message to chat history
            st.session_state["chat_history"].append({"role": "user", "content": user_input})

            # Get response from Gemini API (or placeholder)
            with st.spinner("Getting response..."):
                response = chat_with_gemini(user_input)
            
            # Add assistant response to chat history
            st.session_state["chat_history"].append({"role": "assistant", "content": response})

            # Rerun to update the chat display
            st.rerun()
        else:
            st.warning("Please enter a question to proceed.")

    # Clear chat history button
    if st.button("Clear Chat"):
        st.session_state["chat_history"] = []
        st.rerun()

    st.markdown('</div>', unsafe_allow_html=True)

elif st.session_state["page"] == "ML Accuracy" and st.session_state["logged_in"]:
    st.markdown('<div class="stCard">', unsafe_allow_html=True)
    st.markdown('<div class="card-title">Machine Learning Model Accuracy</div>', unsafe_allow_html=True)
    st.markdown('<div class="card-description">Evaluate the performance of the fall prediction model</div>', unsafe_allow_html=True)

    # Show model accuracy
    try:
        show_model_accuracy()
    except Exception as e:
        st.error(f"Error calculating model accuracy: {e}")

    # Display dataset preview
    st.subheader("Dataset Preview")
    try:
        data = pd.read_csv(data_path)
        st.dataframe(data.head())
    except Exception as e:
        st.error(f"Error loading dataset: {e}")

    st.markdown('</div>', unsafe_allow_html=True)

else:
    if st.session_state["logged_in"]:
        st.warning("Please select a page from the sidebar to continue.")
    else:
        st.warning("Please log in or register to access the application features.")

# Footer
st.markdown("""
<div style="text-align: center; margin-top: 3rem; padding: 1rem; border-top: 1px solid rgba(0, 0, 0, 0.1);">
    <p style="color: #666; font-size: 0.9rem;">
        © 2025 HealthMonitor. All rights reserved. | 
        <a href="#privacy" style="color: #1E88E5; text-decoration: none;">Privacy Policy</a> | 
        <a href="#terms" style="color: #1E88E5; text-decoration: none;">Terms of Service</a>
    </p>
</div>
""", unsafe_allow_html=True)