import os
from pathlib import Path
from flask import Flask, request, jsonify, render_template
import tensorflow as tf
import numpy as np

app = Flask(__name__, static_folder='statics')

# Load the TensorFlow model and scaler in a robust way so the server can start even
# if the artifacts are missing (useful for debugging outside Docker build)
model = None
scaler = None
model_loaded = False
scaler_loaded = False

# Try several plausible locations for the model/scaler so the server works both
# when run from the repo (artifacts next to the repo) and when containerized
# (artifacts copied into /app by dockerfile.serve).
HERE = Path(__file__).resolve().parent
candidate_dirs = [
    HERE.parent,  # repo root (../src -> repo)
    HERE,         # src directory
    Path('/app'), # typical Docker copy target in dockerfile.serve
    Path.cwd(),   # current working dir
    Path('/')     # fallback
]

def find_artifact(name):
    tried = []
    for d in candidate_dirs:
        p = (d / name).resolve()
        tried.append(str(p))
        if p.exists():
            return p, tried
    return None, tried

print(f"[startup] Candidate dirs: {candidate_dirs}")

# Model
MODEL_PATH, model_tried = find_artifact('my_model.keras')
print(f"[startup] Tried model locations: {model_tried}")
try:
    if MODEL_PATH is not None:
        model = tf.keras.models.load_model(str(MODEL_PATH))
        model_loaded = True
        print("[startup] Loaded model from", MODEL_PATH)
    else:
        print("[startup] Model file not found in candidate locations")
except Exception as e:
    model = None
    model_loaded = False
    print("[startup] Failed to load model:", e)

# Scaler
SCALER_PATH, scaler_tried = find_artifact('scaler.pkl')
print(f"[startup] Tried scaler locations: {scaler_tried}")
try:
    import joblib
    if SCALER_PATH is not None:
        scaler = joblib.load(str(SCALER_PATH))
        scaler_loaded = True
        print("[startup] Loaded scaler from", SCALER_PATH)
    else:
        print("[startup] Scaler file not found in candidate locations")
except Exception as e:
    scaler = None
    scaler_loaded = False
    print("[startup] Failed to load scaler:", e)

# Derive class labels from sklearn's wine dataset for clearer output when available
try:
    from sklearn.datasets import load_wine
    class_labels = list(load_wine().target_names)
except Exception:
    # Fallback generic labels
    class_labels = ['Class 0', 'Class 1', 'Class 2']


"""Modern web apps use a technique named routing. This helps the user remember the URLs. 
For instance, instead of having /booking.php they see /booking/. Instead of /account.asp?id=1234/ 
they’d see /account/1234/."""

@app.route('/')
def home():
    return "Welcome to the Iris Classifier API!"

@app.route('/predict', methods=['GET', 'POST'])
def predict():
    if request.method == 'POST':
        try:
            data = request.form
            print("[predict] Received form data:", dict(data))
            # Expecting the first 4 wine features (numeric)
            feature_0 = float(data['feature_0'])
            feature_1 = float(data['feature_1'])
            feature_2 = float(data['feature_2'])
            feature_3 = float(data['feature_3'])

            # Perform the prediction
            input_data = np.array([feature_0, feature_1, feature_2, feature_3])[np.newaxis, ]
            # Apply scaler if present
            if scaler is not None:
                input_data = scaler.transform(input_data)
            if model is None:
                return jsonify({"error": "model not loaded"}), 500
            prediction = model.predict(input_data)
            predicted_class = class_labels[int(np.argmax(prediction))]

            # Return the predicted class in the response
            # Use jsonify() instead of json.dumps() in Flask
            return jsonify({"predicted_class": predicted_class})
        except Exception as e:
            print("[predict] Exception:", e)
            return jsonify({"error": str(e)}), 500
    elif request.method == 'GET':
        return render_template('predict.html')
    else:
        return "Unsupported HTTP method"


@app.route('/health', methods=['GET'])
def health():
    """Simple health endpoint reporting model and scaler availability."""
    return jsonify({
        "status": "ok",
        "model_loaded": model_loaded,
        "scaler_loaded": scaler_loaded
    })

if __name__ == "__main__":
    # Allow the port to be configured via the PORT environment variable (useful for Docker)
    port = int(os.environ.get("PORT", 80))
    app.run(debug=True, host='0.0.0.0', port=port)
