from flask import Flask, request, jsonify
import time
import numpy as np
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import accuracy_score
from Sharp_demo import run_demo

app = Flask(__name__)

@app.route("/health", methods=["GET"])
def health():
    return jsonify({
        "status": "ok",
        "timestamp": time.time(),
        "task_id": sharp_mgr.task_id,
        "model_trained": sharp_mgr.model is not None,
    })

@app.route("/update", methods=["POST"])
def update():
    if not request.is_json:
        return jsonify({"error": "JSON body required"}), 400
    data = request.get_json()
    if "X" not in data or "y" not in data:
        return jsonify({"error": "Both 'X' and 'y' fields are required"}), 400
    try:
        result = sharp_mgr.update(data["X"], data["y"])
    except Exception as e:
        return jsonify({"error": str(e)}), 500
    return jsonify({"success": True, "result": result})

@app.route("/predict", methods=["POST"])
def predict():

    results = run_demo()
    return results

# Optional endpoint to run the original demo script if available
@app.route("/run-demo", methods=["GET"])
def run_demo_endpoint():
    if not HAS_RUN_DEMO:
        return jsonify({"error": "run_demo not available; Sharp_demo.py not importable"}), 500
    try:
        results = run_demo()
        return jsonify({"success": True, "results": results})
    except Exception as e:
        return jsonify({"error": f"Demo failed: {e}"}), 500

if __name__ == "__main__":
    # Run on localhost:8001
    app.run(host="127.0.0.1", port=8001, debug=True)
