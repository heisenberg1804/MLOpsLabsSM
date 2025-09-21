# Dashboard.py

import json
import requests
import streamlit as st
from pathlib import Path
from streamlit.logger import get_logger

# --- PATHS AND ENDPOINTS ---
FASTAPI_BACKEND_ENDPOINT = "http://localhost:8000"
# Make sure your folder structure is correct. This path assumes the script is inside a folder,
# and the FastAPI_Labs folder is two levels up. Adjust if necessary.
FASTAPI_IRIS_MODEL_LOCATION = Path(__file__).resolve().parents[2] / 'FastAPI_Labs' / 'model' / 'iris_model.pkl'
LOGGER = get_logger(__name__)

# --- UI CONFIGURATION AND ASSETS ---
st.set_page_config(
    page_title="Iris Flower Prediction",
    page_icon="🪻",
    layout="wide"
)

# Dictionary to map predictions to flower names and images
IRIS_CLASSES = {
    0: {
        "name": "Setosa",
        "image_url": "../assets/setosa.jpg"
    },
    1: {
        "name": "Versicolor",
        "image_url": "../assets/images.jpeg"
    },
    2: {
        "name": "Virginica",
        "image_url": "../assets/virginica.jpg"
    }
}

# --- HELPER FUNCTIONS ---

def check_backend_status():
    """Checks the FastAPI backend status and displays it in the sidebar."""
    try:
        response = requests.get(FASTAPI_BACKEND_ENDPOINT)
        if response.status_code == 200:
            st.success("Backend online ✅")
        else:
            st.warning(f"Backend returned status code {response.status_code}. 😭")
    except requests.ConnectionError:
        LOGGER.error("Backend offline 😱")
        st.error("Backend offline 😱")

def predict(payload: dict):
    """Sends a prediction request to the backend and returns the response."""
    try:
        response = requests.post(f'{FASTAPI_BACKEND_ENDPOINT}/predict', json=payload)
        response.raise_for_status()  # Raise an exception for bad status codes (4xx or 5xx)
        return response.json()
    except requests.exceptions.RequestException as e:
        LOGGER.error(f"Prediction request failed: {e}")
        st.toast(f':red[Problem connecting to the backend. Please check its status.]', icon="🔴")
        return None

# --- MAIN APP UI ---

def run():
    # --- SIDEBAR ---
    with st.sidebar:
        st.header("Backend Status")
        check_backend_status()
        st.info("This dashboard interacts with a FastAPI backend to predict Iris flower species.")

    # --- MAIN TITLE ---
    st.title("Iris Flower Prediction! 🪻")
    st.markdown("Predict the species of an Iris flower using its measurements.")

    # --- INPUT AND PREDICTION LAYOUT ---
    col1, col2 = st.columns([1, 1]) # Create two columns of equal width

    # --- COLUMN 1: INPUTS ---
    with col1:
        st.header("Input Type")
        input_method = st.selectbox(
            placeholder="Choose your input method",
            options=["Manual Input (Sliders)", "File Upload"],
            label="Select input type"
        )        
        client_input = None # Initialize client_input

        if input_method == "Manual Input (Sliders)":
            st.subheader("Enter Measurements (in cm)")
            sepal_length = st.slider("Sepal Length", 4.3, 7.9, 5.8, 0.1)
            sepal_width = st.slider("Sepal Width", 2.0, 4.4, 3.0, 0.1)
            petal_length = st.slider("Petal Length", 1.0, 6.9, 4.3, 0.1)
            petal_width = st.slider("Petal Width", 0.1, 2.5, 1.3, 0.1)
            
            # Prepare data from sliders for the API
            client_input = {
                "petal_length": petal_length,
                "sepal_length": sepal_length,
                "petal_width": petal_width,
                "sepal_width": sepal_width
            }

        elif input_method == "File Upload":
            st.subheader("Upload a JSON File")
            test_input_file = st.file_uploader(
                'Limit 200KB per file • JSON',
                type=['json']
            )
            
            if test_input_file:
                try:
                    test_input_data = json.load(test_input_file)
                    st.json(test_input_data, expanded=False) # Show a preview of the file
                    # Ensure the file has the expected structure
                    if 'input_test' in test_input_data and isinstance(test_input_data['input_test'], dict):
                         client_input = test_input_data['input_test']
                    else:
                        st.warning("Invalid JSON format. The file must contain an 'input_test' key with the feature dictionary.")
                        client_input = None
                except json.JSONDecodeError:
                    st.error("Invalid JSON file. Please upload a correctly formatted file.")
                    client_input = None
        
        # The predict button
        predict_button = st.button('Predict', type="primary", use_container_width=True)

    # --- COLUMN 2: PREDICTION OUTPUT ---
    with col2:
        st.header("Prediction")
        result_container = st.container(height=350, border=True)

        if predict_button:
            if client_input is None:
                result_container.warning("Please provide valid input before predicting.")
            elif not FASTAPI_IRIS_MODEL_LOCATION.is_file():
                result_container.error("Model file not found. Please ensure `iris_model.pkl` is in the correct location.")
                LOGGER.warning('iris_model.pkl not found. Make sure to run train.py.')
            else:
                with st.spinner('Predicting...'):
                    prediction_response = predict(client_input)
                
                if prediction_response:
                    prediction_val = prediction_response.get("response")
                    
                    if prediction_val in IRIS_CLASSES:
                        flower = IRIS_CLASSES[prediction_val]
                        result_container.success(f"The model predicts: **{flower['name']}**")
                        result_container.image(
                            flower['image_url'],
                            caption=f"Iris {flower['name']}",
                            use_column_width=True
                        )
                    else:
                        result_container.error("An unexpected prediction value was received from the model.")
                        LOGGER.error(f"Unexpected prediction: {prediction_val}")
                else:
                    result_container.error("Failed to get a prediction from the backend.")
        else:
            result_container.info("Click 'Predict' to see the result.")


if __name__ == "__main__":
    run()