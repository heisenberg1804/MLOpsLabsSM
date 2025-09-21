import requests
import json

url = 'http://127.0.0.1:8080/predict'

data = {
    'sepal_length': 5.1,
    'sepal_width': 3.5,
    'petal_length': 1.4,
    'petal_width': 0.2
}

# Add headers and send JSON data
headers = {'Content-Type': 'application/json'}
response = requests.post(url, json=data, headers=headers)

if response.status_code == 200:
    result = response.json()
    print('Prediction Results:')
    print(f"Species: {result['prediction']}")
    print(f"Status: {result['status']}")
else:
    print(f'Error {response.status_code}:', response.text)