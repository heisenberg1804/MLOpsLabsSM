from flask import Flask, request, jsonify
from predict import predict_iris
import os

app = Flask(__name__)

@app.route('/predict', methods=['POST'])
def predict():
    """Endpoint to predict iris species"""
    try:
        data = request.get_json()  # Get data as JSON
        sepal_length = float(data['sepal_length'])
        sepal_width = float(data['sepal_width'])
        petal_length = float(data['petal_length'])
        petal_width = float(data['petal_width'])

        # Get prediction (now returns string species name)
        prediction = predict_iris(sepal_length, sepal_width, petal_length, petal_width)
        
        return jsonify({
            'status': 'success',
            'prediction': prediction,
            'input_features': {
                'sepal_length': sepal_length,
                'sepal_width': sepal_width,
                'petal_length': petal_length,
                'petal_width': petal_width
            }
        })
    
    except Exception as e:
        return jsonify({
            'status': 'error',
            'message': str(e)
        }), 400

@app.route('/health', methods=['GET'])
def health():
    """Health check endpoint"""
    return jsonify({'status': 'healthy'})

if __name__ == '__main__':
    app.run(debug=True, host="0.0.0.0", port=int(os.environ.get("PORT", 8080)))