import streamlit as st
import requests
from sklearn.datasets import load_wine
import numpy as np
from pathlib import Path
from PIL import Image

st.set_page_config(page_title="Wine Classifier", layout="wide")

# Load wine dataset metadata to tune sliders
wine = load_wine()
feature_names = wine.feature_names[:4]
X = wine.data[:, :4]

st.title("Wine Classifier Dashboard")
st.markdown("""
Interactive dashboard for the Wine dataset.
This app calls the local Flask prediction server at http://localhost:4000/predict.
Place class images in `Labs/Docker_Labs/Lab2/src/statics` named `class0.jpeg`, `class1.jpeg`, `class2.jpeg`.
""")

col1, col2 = st.columns([2, 1])

with col1:
    st.header("Input features")
    mins = X.min(axis=0)
    maxs = X.max(axis=0)
    means = X.mean(axis=0)

    f0 = st.slider(f"{feature_names[0]} (feature_0)", float(mins[0]), float(maxs[0]), float(means[0]), step=0.01)
    f1 = st.slider(f"{feature_names[1]} (feature_1)", float(mins[1]), float(maxs[1]), float(means[1]), step=0.01)
    f2 = st.slider(f"{feature_names[2]} (feature_2)", float(mins[2]), float(maxs[2]), float(means[2]), step=0.01)
    f3 = st.slider(f"{feature_names[3]} (feature_3)", float(mins[3]), float(maxs[3]), float(means[3]), step=0.01)

    btn_col1, btn_col2 = st.columns(2)
    with btn_col1:
        if st.button("Predict"):
            payload = {
                'feature_0': float(f0),
                'feature_1': float(f1),
                'feature_2': float(f2),
                'feature_3': float(f3),
            }
            try:
                resp = requests.post('http://localhost:4000/predict', data=payload, timeout=5)
                resp.raise_for_status()
                data = resp.json()
                if 'error' in data:
                    st.error(f"Server error: {data['error']}")
                else:
                    st.session_state['latest_prediction'] = data.get('predicted_class')
                    st.success(f"Predicted class: {st.session_state['latest_prediction']}")
            except Exception as e:
                st.error(f"Prediction request failed: {e}")

    with btn_col2:
        if st.button("Reset"):
            st.experimental_rerun()

    # Single result area (image + prediction)
    latest = st.session_state.get('latest_prediction')
    if latest:
        st.markdown(f"### Latest prediction: {latest}")
        base = Path(__file__).resolve().parent
        img_map = {'class_0': 'class0.jpeg', 'class_1': 'class1.jpeg', 'class_2': 'class2.jpeg'}
        fname = img_map.get(latest, 'class0.jpeg')
        img_path = base / 'statics' / fname
        if img_path.exists():
            try:
                img = Image.open(img_path)
                st.image(img, width=400)
            except Exception as e:
                st.error(f"Failed to load image: {e}")
        else:
            st.info(f"Image not found at {img_path}. Place class images in {base / 'statics'}")

st.caption("Ensure the Flask server is running at http://localhost:4000 for predictions.")
