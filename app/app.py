from flask import Flask, request, jsonify
from sklearn.datasets import load_iris
from sklearn.ensemble import RandomForestClassifier
from prometheus_client import Counter, Histogram, generate_latest
import time

app = Flask(__name__)

# Metrics
REQUEST_COUNT = Counter('request_count', 'App Request Count', ['method', 'endpoint', 'http_status'])
REQUEST_LATENCY = Histogram('request_latency_seconds', 'Request latency', ['endpoint'])

# Model
iris = load_iris()
model = RandomForestClassifier()
model.fit(iris.data, iris.target)

@app.route('/predict', methods=['POST'])
def predict():
    start_time = time.time()
    data = request.json
    features = data.get('features')
    prediction = model.predict([features]).tolist()
    status = 200
    REQUEST_COUNT.labels(method='POST', endpoint='/predict', http_status=status).inc()
    REQUEST_LATENCY.labels(endpoint='/predict').observe(time.time() - start_time)
    return jsonify({'prediction': prediction})

@app.route('/metrics')
def metrics():
    return generate_latest(), 200, {'Content-Type': 'text/plain; version=0.0.4; charset=utf-8'}

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)
