"""
Enhanced Streamlit app to classify uploaded vehicle images with beautiful UI
"""

import streamlit as st
import numpy as np
import tensorflow as tf
import pickle
from PIL import Image
import plotly.express as px
import plotly.graph_objects as go
import pandas as pd

# Page configuration
st.set_page_config(
    page_title="🚗 Vehicle Type Classifier",
    page_icon="🚗",
    layout="centered",
    initial_sidebar_state="collapsed"
)

# Custom CSS for beautiful styling
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap');
    
    .main {
        padding-top: 2rem;
    }
    
    .stApp {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        font-family: 'Inter', sans-serif;
    }
    
    .main-header {
        text-align: center;
        padding: 2rem 0;
        background: linear-gradient(135deg, rgba(255,255,255,0.95) 0%, rgba(255,255,255,0.9) 100%);
        border-radius: 20px;
        margin-bottom: 2rem;
        box-shadow: 0 20px 40px rgba(0,0,0,0.1);
        backdrop-filter: blur(20px);
    }
    
    .main-title {
        font-size: 3rem;
        font-weight: 700;
        background: linear-gradient(135deg, #4facfe 0%, #00f2fe 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        margin-bottom: 0.5rem;
    }
    
    .main-subtitle {
        font-size: 1.2rem;
        color: #666;
        font-weight: 400;
    }
    
    .upload-section {
        background: rgba(255,255,255,0.95);
        padding: 2rem;
        border-radius: 20px;
        margin: 2rem 0;
        box-shadow: 0 15px 35px rgba(0,0,0,0.1);
        backdrop-filter: blur(20px);
    }
    
    .prediction-card {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
        padding: 2rem;
        border-radius: 20px;
        text-align: center;
        margin: 2rem 0;
        box-shadow: 0 15px 35px rgba(102, 126, 234, 0.3);
    }
    
    .prediction-label {
        font-size: 0.9rem;
        opacity: 0.9;
        margin-bottom: 0.5rem;
        text-transform: uppercase;
        letter-spacing: 1px;
    }
    
    .prediction-value {
        font-size: 2.5rem;
        font-weight: 700;
        margin: 0;
        text-shadow: 0 2px 10px rgba(0,0,0,0.2);
    }
    
    .confidence-section {
        background: rgba(255,255,255,0.95);
        padding: 2rem;
        border-radius: 20px;
        margin: 2rem 0;
        box-shadow: 0 10px 30px rgba(0,0,0,0.1);
        backdrop-filter: blur(20px);
    }
    
    .metric-card {
        background: linear-gradient(135deg, #4facfe 0%, #00f2fe 100%);
        color: white;
        padding: 1.5rem;
        border-radius: 15px;
        text-align: center;
        margin: 1rem 0;
        box-shadow: 0 8px 25px rgba(79, 172, 254, 0.3);
    }
    
    .stFileUploader > div > div > div {
        background: linear-gradient(145deg, #f8faff, #e8f4ff);
        border: 3px dashed #4facfe;
        border-radius: 16px;
        padding: 2rem;
    }
    
    .stImage > div {
        border-radius: 16px;
        overflow: hidden;
        box-shadow: 0 10px 30px rgba(0,0,0,0.15);
    }
    
    .success-message {
        background: linear-gradient(135deg, #56ab2f 0%, #a8e6cf 100%);
        color: white;
        padding: 1rem;
        border-radius: 10px;
        text-align: center;
        margin: 1rem 0;
    }
    
    .info-box {
        background: rgba(255,255,255,0.9);
        padding: 1.5rem;
        border-radius: 15px;
        border-left: 5px solid #4facfe;
        margin: 1rem 0;
    }
    
    /* Hide Streamlit branding */
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
    header {visibility: hidden;}
    
    /* Animation for elements */
    .main-header, .upload-section, .prediction-card, .confidence-section {
        animation: slideUp 0.6s ease-out;
    }
    
    @keyframes slideUp {
        from {
            opacity: 0;
            transform: translateY(30px);
        }
        to {
            opacity: 1;
            transform: translateY(0);
        }
    }
</style>
""", unsafe_allow_html=True)

# Header section
st.markdown("""
<div class="main-header">
    <h1 class="main-title">🚗 Vehicle Classifier</h1>
    <p class="main-subtitle">AI-powered vehicle type detection from CCTV images</p>
</div>
""", unsafe_allow_html=True)

@st.cache_resource
def load_model():
    """Load the trained model and label encoder"""
    try:
        model = tf.keras.models.load_model("model/vehicle_cnn_model.h5")
        with open("model/label_encoder.pkl", "rb") as f:
            label_encoder = pickle.load(f)
        return model, label_encoder
    except Exception as e:
        st.error(f"Error loading model: {str(e)}")
        return None, None

def preprocess_image(image):
    """Preprocess the uploaded image for model prediction"""
    image = image.resize((64, 64))
    image = np.array(image) / 255.0
    return np.expand_dims(image, axis=0)

def create_confidence_chart(predictions, labels):
    """Create an interactive confidence chart"""
    df = pd.DataFrame({
        'Vehicle Type': labels,
        'Confidence': predictions * 100
    }).sort_values('Confidence', ascending=True)
    
    fig = px.bar(
        df, 
        x='Confidence', 
        y='Vehicle Type',
        orientation='h',
        color='Confidence',
        color_continuous_scale=['#ff6b6b', '#4ecdc4', '#45b7d1', '#96ceb4', '#ffeaa7'],
        title="🧠 AI Confidence Levels",
        labels={'Confidence': 'Confidence (%)', 'Vehicle Type': 'Vehicle Type'}
    )
    
    fig.update_layout(
        plot_bgcolor='rgba(0,0,0,0)',
        paper_bgcolor='rgba(0,0,0,0)',
        font=dict(family="Inter", size=12),
        title_font=dict(size=16, color='#333'),
        coloraxis_showscale=False,
        height=400
    )
    
    fig.update_traces(
        texttemplate='%{x:.1f}%',
        textposition='inside',
        textfont_color='white',
        textfont_weight='bold'
    )
    
    return fig

# Load model
with st.spinner("🔄 Loading AI model..."):
    model, label_encoder = load_model()

if model is None or label_encoder is None:
    st.error("❌ Unable to load the model. Please check if the model files exist in the 'model' directory.")
    st.info("📁 Expected files:\n- model/vehicle_cnn_model.h5\n- model/label_encoder.pkl")
    st.stop()

# File upload section
st.markdown('<div class="upload-section">', unsafe_allow_html=True)
st.markdown("### 📸 Upload Vehicle Image")

uploaded_file = st.file_uploader(
    "Choose a vehicle image...",
    type=["jpg", "jpeg", "png"],
    help="Upload a clear image of a vehicle for classification"
)

if uploaded_file is not None:
    # Display success message
    st.markdown("""
    <div class="success-message">
        ✅ Image uploaded successfully!
    </div>
    """, unsafe_allow_html=True)
    
    # Process the image
    try:
        image = Image.open(uploaded_file).convert("RGB")
        
        # Create two columns for image and info
        col1, col2 = st.columns([2, 1])
        
        with col1:
            st.image(
                image, 
                caption="📷 Uploaded Image", 
                use_container_width=True
            )
        
        with col2:
            st.markdown("""
            <div class="info-box">
                <h4>🔍 Image Analysis</h4>
                <p><strong>Format:</strong> {}</p>
                <p><strong>Size:</strong> {} x {}</p>
                <p><strong>Mode:</strong> {}</p>
            </div>
            """.format(
                uploaded_file.type.split('/')[-1].upper(),
                image.size[0],
                image.size[1],
                image.mode
            ), unsafe_allow_html=True)
        
        st.markdown('</div>', unsafe_allow_html=True)
        
        # Make prediction
        with st.spinner("🤖 Analyzing image..."):
            input_tensor = preprocess_image(image)
            prediction = model.predict(input_tensor, verbose=0)[0]
            predicted_label = label_encoder.inverse_transform([np.argmax(prediction)])[0]
            confidence = np.max(prediction) * 100
        
        # Display prediction result
        st.markdown(f"""
        <div class="prediction-card">
            <div class="prediction-label">Detected Vehicle Type</div>
            <h1 class="prediction-value">{predicted_label.upper()}</h1>
            <p style="margin-top: 1rem; font-size: 1.1rem;">
                Confidence: {confidence:.1f}%
            </p>
        </div>
        """, unsafe_allow_html=True)
        
        # Display confidence chart
        st.markdown('<div class="confidence-section">', unsafe_allow_html=True)
        
        all_labels = label_encoder.classes_
        fig = create_confidence_chart(prediction, all_labels)
        st.plotly_chart(fig, use_container_width=True)
        
        # Additional metrics
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.markdown(f"""
            <div class="metric-card">
                <h3>{confidence:.1f}%</h3>
                <p>Top Confidence</p>
            </div>
            """, unsafe_allow_html=True)
        
        with col2:
            second_highest = np.partition(prediction, -2)[-2] * 100
            st.markdown(f"""
            <div class="metric-card">
                <h3>{second_highest:.1f}%</h3>
                <p>2nd Highest</p>
            </div>
            """, unsafe_allow_html=True)
        
        with col3:
            certainty = (confidence - second_highest)
            st.markdown(f"""
            <div class="metric-card">
                <h3>{certainty:.1f}%</h3>
                <p>Certainty Gap</p>
            </div>
            """, unsafe_allow_html=True)
        
        st.markdown('</div>', unsafe_allow_html=True)
        
        # Detailed results in expander
        with st.expander("📊 Detailed Classification Results"):
            results_df = pd.DataFrame({
                'Vehicle Type': all_labels,
                'Confidence (%)': prediction * 100,
                'Probability': prediction
            }).sort_values('Confidence (%)', ascending=False)
            
            st.dataframe(
                results_df,
                use_container_width=True,
                hide_index=True
            )
    
    except Exception as e:
        st.error(f"❌ Error processing image: {str(e)}")

else:
    st.markdown('</div>', unsafe_allow_html=True)
    
    # Information section when no file is uploaded
    st.markdown("""
    <div class="info-box">
        <h4>ℹ️ How to use this classifier:</h4>
        <ol>
            <li>Upload a clear image of a vehicle (JPG, JPEG, or PNG format)</li>
            <li>Wait for the AI to analyze the image</li>
            <li>View the classification results and confidence levels</li>
            <li>Explore detailed breakdown of all vehicle type probabilities</li>
        </ol>
        <p><strong>Tip:</strong> For best results, use high-quality images with the vehicle clearly visible.</p>
    </div>
    """, unsafe_allow_html=True)

# Footer
st.markdown("---")
st.markdown("""
<div style="text-align: center; color: rgba(255,255,255,0.8); padding: 2rem;">
    <p>🚗 Vehicle Classification System | Powered by TensorFlow & Streamlit</p>
    <p style="font-size: 0.8rem;">Upload an image to get started with AI-powered vehicle detection</p>
</div>
""", unsafe_allow_html=True)