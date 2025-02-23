import streamlit as st
import pandas as pd
import numpy as np
import pickle
import plotly.express as px
import plotly.graph_objects as go
from datetime import datetime
import time
import random
from langchain_groq import ChatGroq
from langchain.chains import LLMChain
from langchain.prompts import PromptTemplate
import logging
from typing import Tuple, List
import queue
from dataclasses import dataclass
from enum import Enum
import os
from dotenv import load_dotenv

load_dotenv()
api_key = os.getenv("GROQ_API_KEY")


class ThreatLevel(Enum):
    LOW = "Low"
    MEDIUM = "Medium"
    HIGH = "High"
    CRITICAL = "Critical"

@dataclass
class NetworkMetrics:
    traffic: float
    latency: float
    packet_loss: float
    suspicious_activity: float
    timestamp: float = time.time()

class MetricsQueue:
    def __init__(self, maxsize: int = 1000):
        self.queue = queue.Queue(maxsize)
        
    def put(self, metrics: NetworkMetrics) -> None:
        if self.queue.full():
            self.queue.get()
        self.queue.put(metrics)
        
    def get_all(self) -> List[NetworkMetrics]:
        return list(self.queue.queue)

class NetworkMonitor:
    def __init__(self):
        self.metrics_queue = MetricsQueue()
        self.alert_threshold = {
            'suspicious_activity': 7.0,
            'packet_loss': 3.0
        }
        
    def generate_metrics(self) -> NetworkMetrics:
        return NetworkMetrics(
            traffic=random.uniform(100, 1000),
            latency=random.uniform(1, 100),
            packet_loss=random.uniform(0, 5),
            suspicious_activity=random.uniform(0, 10)
        )
        
    def check_alerts(self, metrics: NetworkMetrics) -> List[str]:
        alerts = []
        if metrics.suspicious_activity > self.alert_threshold['suspicious_activity']:
            alerts.append("⚠️ High suspicious activity detected!")
        if metrics.packet_loss > self.alert_threshold['packet_loss']:
            alerts.append("⚠️ High packet loss detected!")
        return alerts

st.set_page_config(
    page_title="Advanced NIDS",
    page_icon="🛡️",
    layout="wide",
    initial_sidebar_state="expanded"
)

st.markdown("""
    <style>
    .main {
        background-color: #1a1a1a;
        color: #ffffff;
    }
    .stButton>button {
        background-color: #2E7D32;
        color: white;
        border-radius: 20px;
    }
    .stTextInput>div>div>input {
        background-color: #2b2b2b;
        color: white;
    }
    .stSelectbox>div>div>input {
        background-color: #2b2b2b;
        color: white;
    }
    .block-container {
        padding-top: 1rem;
        padding-bottom: 1rem;
    }
    </style>
    """, unsafe_allow_html=True)


@st.cache_resource
def initialize_llm() -> Tuple[LLMChain, LLMChain, LLMChain]:
    try:
        llm = ChatGroq(
            model="qwen-2.5-32b",
            temperature=0.2,
            groq_api_key=api_key,
            max_tokens=2000
        )
        
        prompt1 = PromptTemplate(
            input_variables=["protocol", "packet_size", "flow_duration", "bytes_per_second", 
                           "login_attempts", "api_requests", "abnormal_data_transfer", 
                           "encrypted_traffic_ratio", "failed_auth_attempts", "tcp_flags", 
                           "suspicious_ip_flag", "port_scan_activity", "suspicious_dns_request", 
                           "anomalous_behavior_score"],
            template="""
            Given the following network data:
            Protocol: {protocol}
            Packet Size: {packet_size}
            Flow Duration: {flow_duration}
            Bytes Per Second: {bytes_per_second}
            Login Attempts: {login_attempts}
            API Requests: {api_requests}
            Abnormal Data Transfer: {abnormal_data_transfer} MB
            Encrypted Traffic Ratio: {encrypted_traffic_ratio}
            Failed Authentication Attempts: {failed_auth_attempts}
            TCP Flags: {tcp_flags}
            Suspicious IP Flag: {suspicious_ip_flag}
            Port Scan Activity: {port_scan_activity}
            Suspicious DNS Request: {suspicious_dns_request}
            Anomalous Behavior Score: {anomalous_behavior_score}
            
            Analyze this data and extract key insights about the network activity.
            """
        )

        prompt2 = PromptTemplate(
            input_variables=["network_insights"],
            template="""
            Based on the network insights:
            {network_insights}
            
            Provide a detailed threat analysis addressing potential security concerns.
            """
        )

        prompt3 = PromptTemplate(
            input_variables=["threat_analysis"],
            template="""
            Based on the threat analysis:
            {threat_analysis}
            
            Generate a comprehensive security report with actionable recommendations.
            """
        )

        chain1 = LLMChain(llm=llm, prompt=prompt1, output_key="network_insights")
        chain2 = LLMChain(llm=llm, prompt=prompt2, output_key="threat_analysis")
        chain3 = LLMChain(llm=llm, prompt=prompt3, output_key="security_report")
        
        return chain1, chain2, chain3
    except Exception as e:
        logging.error(f"Error initializing LLM: {str(e)}")
        raise

@st.cache_resource
def load_model() -> Tuple:
    try:
        with open("model.pkl", "rb") as f:
            model_data = pickle.load(f)
        return model_data
    except FileNotFoundError:
        st.error("⚠️ Model file not found. Please ensure model.pkl is in the directory.")
        st.stop()
    except Exception as e:
        st.error(f"⚠️ Error loading model: {str(e)}")
        st.stop()

try:
    logistic_model, scaler, label_encoders, feature_names = load_model()
    chain1, chain2, chain3 = initialize_llm()
except Exception as e:
    st.error(f"⚠️ Error initializing components: {str(e)}")
    st.stop()

with st.sidebar:
    st.title("🛡️ NIDS Control Panel")
    st.markdown("---")
    
    mode = st.selectbox(
        "Select Operation Mode",
        ["Real-time Monitoring", "Advanced Threat Analysis", "System Performance"]
    )
    
    dark_mode = st.toggle("Dark Mode", value=True)
    enable_llm = st.toggle("Enable LLM Analysis", value=True)
    
    st.markdown("---")
    st.markdown("### System Status")
    col1, col2 = st.columns(2)
    with col1:
        current_connections = random.randint(50, 200)
        st.metric("Active Connections", current_connections, 
                 delta=random.randint(-10, 10))
    with col2:
        threat_level = random.choice(list(ThreatLevel))
        st.metric("Threat Level", threat_level.value)

st.title("🌐 Network Intrusion Detection System")

if mode == "Real-time Monitoring":
    st.subheader("Real-time Network Traffic Monitoring")
    
    if 'network_monitor' not in st.session_state:
        st.session_state.network_monitor = NetworkMonitor()
    
    col1, col2, col3 = st.columns(3)
    with col1:
        monitoring_active = st.toggle("Active Monitoring", value=True)
    with col2:
        update_interval = st.slider("Update Interval (seconds)", 1, 10, 2)
    with col3:
        max_data_points = st.slider("History Length (points)", 10, 100, 50)
    
    metrics_placeholder = st.empty()
    chart_placeholder = st.empty()
    alert_placeholder = st.empty()
    
    if monitoring_active:
        metrics = st.session_state.network_monitor.generate_metrics()
        st.session_state.network_monitor.metrics_queue.put(metrics)
        
        metrics_data = st.session_state.network_monitor.metrics_queue.get_all()
        df = pd.DataFrame([vars(m) for m in metrics_data])
        
        with metrics_placeholder.container():
            m1, m2, m3, m4 = st.columns(4)
            
            with m1:
                st.metric(
                    "Network Traffic",
                    f"{metrics.traffic:.1f} Mbps",
                    delta=f"{metrics.traffic - df['traffic'].mean():.1f}"
                )
            
            with m2:
                st.metric(
                    "Latency",
                    f"{metrics.latency:.1f} ms",
                    delta=f"{metrics.latency - df['latency'].mean():.1f}"
                )
            
            with m3:
                st.metric(
                    "Packet Loss",
                    f"{metrics.packet_loss:.2f}%",
                    delta=f"{metrics.packet_loss - df['packet_loss'].mean():.2f}"
                )
            
            with m4:
                st.metric(
                    "Suspicious Activity",
                    f"{metrics.suspicious_activity:.1f}",
                    delta=f"{metrics.suspicious_activity - df['suspicious_activity'].mean():.1f}"
                )
        
        with chart_placeholder.container():
            c1, c2 = st.columns(2)
            
            with c1:
                fig1 = go.Figure()
                fig1.add_trace(go.Scatter(
                    y=df['traffic'],
                    name='Traffic (Mbps)',
                    line=dict(color='#2E7D32')
                ))
                fig1.add_trace(go.Scatter(
                    y=df['latency'],
                    name='Latency (ms)',
                    line=dict(color='#FFA726')
                ))
                fig1.update_layout(
                    title='Network Traffic and Latency',
                    height=300,
                    margin=dict(l=0, r=0, t=30, b=0),
                    template='plotly_dark' if dark_mode else 'plotly'
                )
                st.plotly_chart(fig1, use_container_width=True)
            
            with c2:
                fig2 = go.Figure()
                fig2.add_trace(go.Scatter(
                    y=df['packet_loss'],
                    name='Packet Loss (%)',
                    line=dict(color='#EF5350')
                ))
                fig2.add_trace(go.Scatter(
                    y=df['suspicious_activity'],
                    name='Suspicious Activity',
                    line=dict(color='#AB47BC')
                ))
                fig2.update_layout(
                    title='Packet Loss and Suspicious Activity',
                    height=300,
                    margin=dict(l=0, r=0, t=30, b=0),
                    template='plotly_dark' if dark_mode else 'plotly'
                )
                st.plotly_chart(fig2, use_container_width=True)
        
        alerts = st.session_state.network_monitor.check_alerts(metrics)
        with alert_placeholder.container():
            for alert in alerts:
                if 'suspicious activity' in alert.lower():
                    st.error(alert)
                else:
                    st.warning(alert)
        
        time.sleep(update_interval)
        st.rerun()

elif mode == "Advanced Threat Analysis":
    st.subheader("Network Traffic Analysis with LLM Integration")
    
    col1, col2 = st.columns(2)
    
    with col1:
        protocol = st.selectbox("Protocol", label_encoders['Protocol'].classes_)
        packet_size = st.slider("Packet Size", 0, 1000, 500)
        flow_duration = st.slider("Flow Duration", 0, 1000, 100)
        bytes_per_second = st.slider("Bytes Per Second", 0.0, 1000.0, 100.0)
        login_attempts = st.number_input("Login Attempts", 0, 100, 10)
        api_requests = st.number_input("API Requests", 0, 1000, 100)
        abnormal_data_transfer = st.slider("Abnormal Data Transfer (MB)", 0.0, 100.0, 10.0)
    
    with col2:
        encrypted_traffic_ratio = st.slider("Encrypted Traffic Ratio", 0.0, 1.0, 0.5)
        failed_auth_attempts = st.number_input("Failed Authentication Attempts", 0, 100, 5)
        tcp_flags = st.selectbox("TCP Flags", label_encoders['TCP_Flags'].classes_)
        suspicious_ip_flag = st.selectbox("Suspicious IP Flag", [0, 1])
        port_scan_activity = st.number_input("Port Scan Activity", 0, 100, 20)
        suspicious_dns_request = st.number_input("Suspicious DNS Request", 0, 100, 15)
        anomalous_behavior_score = st.slider("Anomalous Behavior Score", 0.0, 1.0, 0.5)
    
    if st.button("Analyze Traffic", key="analyze"):
        with st.spinner("Analyzing network traffic..."):
            try:
                # Create DataFrame with exact feature names
                input_data = pd.DataFrame({
                    'Protocol': [protocol],
                    'TCP_Flags': [tcp_flags],
                    'Packet_Size': [packet_size],
                    'Flow_Duration': [flow_duration],
                    'Bytes_Per_Second': [bytes_per_second],
                    'Login_Attempts': [login_attempts],
                    'API_Requests': [api_requests],
                    'Abnormal_Data_Transfer_MB': [abnormal_data_transfer],
                    'Encrypted_Traffic_Ratio': [encrypted_traffic_ratio],
                    'Failed_Auth_Attempts': [failed_auth_attempts],
                    'Suspicious_IP_Flag': [suspicious_ip_flag],
                    'Port_Scan_Activity': [port_scan_activity],
                    'Suspicious_DNS_Request': [suspicious_dns_request],
                    'Anomalous_Behavior_Score': [anomalous_behavior_score]
                })
                
                # Transform categorical variables first
                input_data['Protocol'] = label_encoders['Protocol'].transform(input_data['Protocol'])
                input_data['TCP_Flags'] = label_encoders['TCP_Flags'].transform(input_data['TCP_Flags'])
                
                # Ensure column order matches feature_names
                input_data = input_data[feature_names]
                
                # Scale numeric features
                numeric_cols = input_data.select_dtypes(include=['int64', 'float64']).columns
                input_data[numeric_cols] = scaler.transform(input_data[numeric_cols])
                
                # Make prediction
                prediction = logistic_model.predict(input_data)
                prediction_proba = logistic_model.predict_proba(input_data)
                
                result = label_encoders['Object_Type'].inverse_transform(prediction)[0]
                confidence = np.max(prediction_proba) * 100
                
                # Display results
                st.markdown("### 🎯 ML Model Results")
                cols = st.columns(3)
                
                with cols[0]:
                    st.markdown(f"""
                        **Detected Threat**: {result}  
                        **Confidence**: {confidence:.2f}%  
                        **Analysis Time**: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
                    """)
                
                with cols[1]:
                    fig = go.Figure(go.Indicator(
                        mode="gauge+number",
                        value=confidence,
                        title={'text': "Confidence Score"},
                        gauge={
                            'axis': {'range': [0, 100]},
                            'bar': {'color': "darkgreen"},
                            'steps': [
                                {'range': [0, 50], 'color': "lightgray"},
                                {'range': [50, 75], 'color': "gray"},
                                {'range': [75, 100], 'color': "darkgray"}
                            ]
                        }
                    ))
                    fig.update_layout(height=300, template='plotly_dark' if dark_mode else 'plotly')
                    st.plotly_chart(fig, use_container_width=True)
                
                with cols[2]:
                    threat_types = label_encoders['Object_Type'].classes_
                    fig = px.bar(
                        x=threat_types,
                        y=prediction_proba[0],
                        title="Threat Probability Distribution"
                    )
                    fig.update_layout(
                        xaxis_title="Threat Type",
                        yaxis_title="Probability",
                        height=300,
                        template='plotly_dark' if dark_mode else 'plotly'
                    )
                    st.plotly_chart(fig, use_container_width=True)
                
                if enable_llm:
                    try:
                        # Convert input data to format expected by LLM
                        llm_input = {
                            'protocol': protocol,
                            'packet_size': str(packet_size),
                            'flow_duration': str(flow_duration),
                            'bytes_per_second': str(bytes_per_second),
                            'login_attempts': str(login_attempts),
                            'api_requests': str(api_requests),
                            'abnormal_data_transfer': str(abnormal_data_transfer),
                            'encrypted_traffic_ratio': str(encrypted_traffic_ratio),
                            'failed_auth_attempts': str(failed_auth_attempts),
                            'tcp_flags': tcp_flags,
                            'suspicious_ip_flag': str(suspicious_ip_flag),
                            'port_scan_activity': str(port_scan_activity),
                            'suspicious_dns_request': str(suspicious_dns_request),
                            'anomalous_behavior_score': str(anomalous_behavior_score)
                        }
                        
                        chain_results = chain1.invoke(**llm_input)
                        threat_results = chain2.invoke(network_insights=chain_results)
                        security_report = chain3.invoke(threat_analysis=threat_results)
                        
                        tabs = st.tabs(["Network Insights", "Threat Analysis", "Security Report"])
                        
                        with tabs[0]:
                            st.markdown(chain_results)
                        with tabs[1]:
                            st.markdown(threat_results)
                        with tabs[2]:
                            st.markdown(security_report)
                    
                    except Exception as e:
                        st.error(f"LLM Analysis Error: {str(e)}")
                        st.info("Continuing with statistical analysis only")
            
            except Exception as e:
                st.error(f"Error in prediction: {str(e)}")
                st.error("Debug info:")
                st.write("Feature names expected:", feature_names)
                st.write("Features provided:", input_data.columns.tolist())

elif mode == "System Performance":
    st.subheader("System Performance Monitoring")
    
    # Initialize session state for performance data
    if 'performance_data' not in st.session_state:
        st.session_state.performance_data = []
        st.session_state.last_update = time.time()
    
    # Update interval in seconds
    UPDATE_INTERVAL = 5
    
    # Check if it's time to update
    current_time = time.time()
    if current_time - st.session_state.last_update >= UPDATE_INTERVAL:
        # Generate new performance data
        current_data = {
            'timestamp': datetime.now(),
            'cpu_usage': random.uniform(20, 80),
            'memory_usage': random.uniform(30, 90),
            'disk_usage': random.uniform(40, 95),
            'network_throughput': random.uniform(50, 500),
            'active_connections': random.randint(50, 200),
            'system_load': random.uniform(0, 1),
            'io_wait': random.uniform(0, 20)
        }
        
        st.session_state.performance_data.append(current_data)
        if len(st.session_state.performance_data) > 100:
            st.session_state.performance_data.pop(0)
        
        st.session_state.last_update = current_time
    
    # Display current metrics
    col1, col2, col3 = st.columns(3)
    current_data = st.session_state.performance_data[-1] if st.session_state.performance_data else {
        'cpu_usage': 0, 'memory_usage': 0, 'disk_usage': 0
    }
    
    with col1:
        st.metric(
            "CPU Usage",
            f"{current_data['cpu_usage']:.1f}%",
            delta=f"{random.uniform(-5, 5):.1f}%"
        )
    
    with col2:
        st.metric(
            "Memory Usage",
            f"{current_data['memory_usage']:.1f}%",
            delta=f"{random.uniform(-5, 5):.1f}%"
        )
    
    with col3:
        st.metric(
            "Disk Usage",
            f"{current_data['disk_usage']:.1f}%",
            delta=f"{random.uniform(-2, 2):.1f}%"
        )
    
    # Create DataFrame from performance data
    if st.session_state.performance_data:
        df_performance = pd.DataFrame(st.session_state.performance_data)
        
        # Display tabs
        tab1, tab2 = st.tabs(["Resource Usage", "System Metrics"])
        
        with tab1:
            st.subheader("System Resource Usage")
            fig = go.Figure()
            
            metrics_to_plot = {
                'cpu_usage': {'name': 'CPU Usage', 'color': '#1f77b4'},
                'memory_usage': {'name': 'Memory Usage', 'color': '#ff7f0e'},
                'disk_usage': {'name': 'Disk Usage', 'color': '#2ca02c'}
            }
            
            for metric, config in metrics_to_plot.items():
                fig.add_trace(go.Scatter(
                    x=df_performance['timestamp'],
                    y=df_performance[metric],
                    name=config['name'],
                    line=dict(width=2, color=config['color'])
                ))
            
            fig.update_layout(
                title='System Resource Usage Over Time',
                yaxis_title='Usage (%)',
                height=400,
                template='plotly_dark' if dark_mode else 'plotly',
                legend=dict(
                    orientation="h",
                    yanchor="bottom",
                    y=1.02,
                    xanchor="right",
                    x=1
                )
            )
            
            st.plotly_chart(fig, use_container_width=True)
        
        with tab2:
            st.subheader("Additional System Metrics")
            col1, col2 = st.columns(2)
            
            with col1:
                fig = go.Figure(go.Indicator(
                    mode="gauge+number",
                    value=current_data['system_load'],
                    title={'text': "System Load Average"},
                    gauge={
                        'axis': {'range': [0, 1]},
                        'bar': {'color': "darkgreen"},
                        'steps': [
                            {'range': [0, 0.5], 'color': "lightgray"},
                            {'range': [0.5, 0.8], 'color': "gray"},
                            {'range': [0.8, 1], 'color': "darkgray"}
                        ]
                    }
                ))
                fig.update_layout(height=300, template='plotly_dark' if dark_mode else 'plotly')
                st.plotly_chart(fig, use_container_width=True)
            
            with col2:
                fig = go.Figure(go.Indicator(
                    mode="gauge+number",
                    value=current_data['io_wait'],
                    title={'text': "I/O Wait (%)"},
                    gauge={
                        'axis': {'range': [0, 20]},
                        'bar': {'color': "darkgreen"},
                        'steps': [
                            {'range': [0, 5], 'color': "lightgray"},
                            {'range': [5, 10], 'color': "gray"},
                            {'range': [10, 20], 'color': "darkgray"}
                        ]
                    }
                ))
                fig.update_layout(height=300, template='plotly_dark' if dark_mode else 'plotly')
                st.plotly_chart(fig, use_container_width=True)
            
            # Network metrics
            st.subheader("Network Performance")
            col3, col4 = st.columns(2)
            
            with col3:
                fig = px.line(
                    df_performance,
                    x='timestamp',
                    y='network_throughput',
                    title='Network Throughput'
                )
                fig.update_layout(
                    yaxis_title='Mbps',
                    height=300,
                    template='plotly_dark' if dark_mode else 'plotly'
                )
                st.plotly_chart(fig, use_container_width=True)
            
            with col4:
                fig = px.line(
                    df_performance,
                    x='timestamp',
                    y='active_connections',
                    title='Active Connections'
                )
                fig.update_layout(
                    height=300,
                    template='plotly_dark' if dark_mode else 'plotly'
                )
                st.plotly_chart(fig, use_container_width=True)
            
            # System Health Analysis
            with st.expander("System Health Analysis"):
                health_score = 100 - (
                    current_data['cpu_usage'] +
                    current_data['memory_usage'] +
                    current_data['disk_usage']
                ) / 3
                
                st.markdown(f"""
                    ### System Health Score: {health_score:.1f}%
                    
                    Current Status:
                    - Network Throughput: {current_data['network_throughput']:.1f} Mbps
                    - Active Connections: {current_data['active_connections']}
                    - System Load: {current_data['system_load']:.2f}
                    - I/O Wait: {current_data['io_wait']:.1f}%
                """)
                
                if health_score < 50:
                    st.error("⚠️ System requires immediate attention!")
                elif health_score < 70:
                    st.warning("⚠️ System performance is suboptimal.")
                else:
                    st.success("✅ System is performing optimally.")

# Footer Section
st.markdown("---")
