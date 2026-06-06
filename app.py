from flask import Flask, jsonify, request

app = Flask(__name__)

@app.route('/')
def home():
    return jsonify({
        "status": "Success",
        "message": "ML Model API is live and healthy!"
    })

@app.route('/predict', methods=['POST'])
def predict():
    # Yeh simulate kar raha hai model prediction ko
    return jsonify({
        "prediction": "Iris-Setosa",
        "confidence": "98.4%"
    })

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)
