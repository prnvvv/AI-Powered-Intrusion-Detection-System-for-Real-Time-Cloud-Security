import pickle
import pandas as pd
import numpy as np
import streamlit as st
import seaborn as sns
import matplotlib.pyplot as plt
import ipaddress
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

st.set_page_config(page_title="Cloud NIDS Dashboard", layout="wide")
st.markdown("""
    <style>
        body {background-color: #0e1117; color: #ffffff;}
        .stApp {background-color: #0e1117;}
        .title {text-align: center; font-size: 40px; font-weight: bold; color: #00ffff;}
        .sidebar .sidebar-content {background-color: #14161a;}
    </style>
""", unsafe_allow_html=True)

st.markdown("<h1 class='title'>Cloud Network Intrusion Detection System</h1>", unsafe_allow_html=True)

# Load dataset and Train Model
def load_or_train_model():
    try:
        with open("best_model.pkl", "rb") as f:
            return pickle.load(f)
    except:
        df = pd.read_csv("AI_Cybersecurity_Complete_IDS_Dataset.csv")
        label_encoders = {}
        
        for col in ["Protocol", "TCP_Flags", "Object_Type"]:
            le = LabelEncoder()
            df[col] = le.fit_transform(df[col])
            label_encoders[col] = le
        
        features = [col for col in df.columns if col != "Object_Type"]
        X = df[features]
        y = df["Object_Type"]
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        
        scaler = StandardScaler()
        X_train = scaler.fit_transform(X_train)
        X_test = scaler.transform(X_test)
        
        model = RandomForestClassifier(n_estimators=100, random_state=42)
        model.fit(X_train, y_train)
        
        accuracy = accuracy_score(y_test, model.predict(X_test))
        st.sidebar.write(f"Model Trained with Accuracy: {accuracy:.2f}")
        
        with open("best_model.pkl", "wb") as f:
            pickle.dump((model, scaler, label_encoders), f)
        
        return model, scaler, label_encoders

model, scaler, label_encoders = load_or_train_model()

st.sidebar.title("Navigation")
menu = st.sidebar.radio("Go to", ["Live Prediction", "Exploratory Data Analysis"])

if menu == "Live Prediction":
    st.subheader("Enter Network Data for Intrusion Detection")
    
    col1, col2 = st.columns(2)
    
    with col1:
        source_ip = st.text_input("Source IP")
        dest_ip = st.text_input("Destination IP")
        protocol = st.selectbox("Protocol", label_encoders["Protocol"].classes_)
        packet_size = st.number_input("Packet Size", min_value=0, max_value=2000, step=1)
        flow_duration = st.number_input("Flow Duration", step=1)
        bytes_per_second = st.number_input("Bytes Per Second", format="%.5f")
        login_attempts = st.number_input("Login Attempts", step=1)
        api_requests = st.number_input("API Requests", step=1)
    
    with col2:
        abnormal_transfer = st.number_input("Abnormal Data Transfer (MB)", step=1)
        encrypted_ratio = st.number_input("Encrypted Traffic Ratio", format="%.5f")
        failed_auth = st.number_input("Failed Authentication Attempts", step=1)
        tcp_flags = st.selectbox("TCP Flags", label_encoders["TCP_Flags"].classes_)
        suspicious_ip = st.radio("Suspicious IP Flag", ["No", "Yes"])
        port_scan_activity = st.radio("Port Scan Activity", ["No", "Yes"])
        suspicious_dns_request = st.radio("Suspicious DNS Request", ["No", "Yes"])
        anomalous_behavior_score = st.number_input("Anomalous Behavior Score", format="%.5f")
    
    if st.button("Predict Intrusion Type"):
        def ip_to_int(ip):
            try:
                return int(ipaddress.IPv4Address(ip))
            except ValueError:
                return 0
        
        user_data = pd.DataFrame([{
            "Source_IP": ip_to_int(source_ip),
            "Destination_IP": ip_to_int(dest_ip),
            "Protocol": label_encoders["Protocol"].transform([protocol])[0],
            "Packet_Size": int(packet_size),
            "Flow_Duration": int(flow_duration),
            "Bytes_Per_Second": float(bytes_per_second),
            "Login_Attempts": int(login_attempts),
            "API_Requests": int(api_requests),
            "Abnormal_Data_Transfer_MB": int(abnormal_transfer),
            "Encrypted_Traffic_Ratio": float(encrypted_ratio),
            "Failed_Auth_Attempts": int(failed_auth),
            "TCP_Flags": label_encoders["TCP_Flags"].transform([tcp_flags])[0],
            "Suspicious_IP_Flag": 1 if suspicious_ip == "Yes" else 0,
            "Port_Scan_Activity": 1 if port_scan_activity == "Yes" else 0,
            "Suspicious_DNS_Request": 1 if suspicious_dns_request == "Yes" else 0,
            "Anomalous_Behavior_Score": float(anomalous_behavior_score)
        }])
        
        scaled_data = scaler.transform(user_data)
        prediction = model.predict(scaled_data)
        predicted_label = label_encoders["Object_Type"].inverse_transform(prediction)[0]
        st.success(f"Predicted Intrusion Type: {predicted_label}")

elif menu == "Exploratory Data Analysis":
    st.subheader("Data Visualization & Insights")
    df = pd.read_csv("AI_Cybersecurity_Complete_IDS_Dataset.csv")
    numeric_features = df.select_dtypes(include=["int", "float"]).columns.tolist()
    
    st.write("### Distribution of Attack Types")
    fig, ax = plt.subplots()
    sns.countplot(x=df["Object_Type"], palette="coolwarm", ax=ax)
    st.pyplot(fig)
    
    st.write("### Correlation Heatmap")
    fig, ax = plt.subplots(figsize=(10, 6))
    sns.heatmap(df[numeric_features].corr(), annot=True, cmap="coolwarm", fmt=".2f", ax=ax)
    st.pyplot(fig)
    
    st.write("### Feature Distributions")
    selected_feature = st.selectbox("Select Feature", numeric_features)
    fig, ax = plt.subplots()
    sns.histplot(df[selected_feature], kde=True, bins=30, ax=ax)
    st.pyplot(fig)
