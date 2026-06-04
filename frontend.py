import streamlit as st
import requests

# 1. Page Configuration
st.set_page_config(page_title="AI Diagnostic Platform", page_icon="🩺", layout="centered")

st.title("🩺 AI Diagnostic Platform")
st.markdown("Upload a medical scan to receive an instant, AI-powered diagnostic prediction.")

# 2. User Inputs
scan_type = st.selectbox("Select Scan Type:", ["Brain Tumor (MRI)", "Ovarian Cancer (Ultrasound/CT)"])
uploaded_file = st.file_uploader("Upload Medical Image", type=["jpg", "png", "jpeg"])

# 3. Process the Image
if uploaded_file is not None:
    st.image(uploaded_file, caption="Uploaded Scan", use_container_width=True)
    
    if st.button("Run AI Diagnostics", type="primary"):
        with st.spinner("Processing image through neural network..."):
            
            # Determine which FastAPI endpoint to hit
            endpoint = "brain-tumor" if "Brain" in scan_type else "ovarian-cancer"
            api_url = f"https://full-stack-ai-diagnostic-data-workflow.onrender.com/predict/{endpoint}/"
            
            # Package the file to send to the backend
            files = {"file": (uploaded_file.name, uploaded_file.getvalue(), uploaded_file.type)}
            
            try:
                # Send the POST request to FastAPI
                response = requests.post(api_url, files=files)
                
                if response.status_code == 200:
                    data = response.json()
                    
                    # 4. Display the Results Beautifully
                    st.divider()
                    st.subheader("Diagnostic Results")
                    
                    prediction = data.get("prediction")
                    confidence_str = data.get("confidence", "0%")
                    confidence_val = float(confidence_str.strip("%"))
                    
                    # Color-coded success/warning boxes
                    if prediction == "Malignant":
                        st.error(f"**Prediction:** {prediction}")
                    else:
                        st.success(f"**Prediction:** {prediction}")
                        
                    st.metric("Confidence Score", confidence_str)
                    st.progress(confidence_val / 100) # Progress bar
                    
                    # Hidden accordion for the raw data
                    with st.expander("View Raw Backend JSON Data"):
                        st.json(data)
                else:
                    st.error("API Error. Please check your backend connection.")
                    
            except requests.exceptions.ConnectionError:
                st.error("🚨 Connection Error: Cannot reach the FastAPI backend. Is it running on port 8000?")
