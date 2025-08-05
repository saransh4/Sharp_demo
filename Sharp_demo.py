# Install required packages (run this cell first)
# !pip install scikit-learn numpy

# SHARP Demo Code - Simple Version for Google Colab
import numpy as np
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import accuracy_score
import time

def sharp_demo(previous_model, new_data, task_id):
    """
    SHARP Demo: model[k-1] + new_data -> model[k]
    
    Args:
        previous_model: Previous trained model (or None for first task)
        new_data: Tuple of (X, y) training data
        task_id: Current task number
    
    Returns:
        Dict with new model and performance info
    """
    X_new, y_new = new_data
    
    print(f"🚀 Processing Task {task_id} with {len(X_new)} samples")
    
    # Create or adapt model
    if previous_model is None:
        # First task - create new model
        model = MLPClassifier(hidden_layer_sizes=(100, 50), max_iter=300, random_state=42)
        adaptation_type = "Initial Training"
    else:
        # Subsequent tasks - use previous model with warm start
        model = previous_model  # Reuse the same model object
        model.max_iter += 100   # Add more iterations for new data
        adaptation_type = "Incremental Learning"
    
    # Train the model
    start_time = time.time()
    model.fit(X_new, y_new)
    training_time = time.time() - start_time
    
    # Evaluate performance
    y_pred = model.predict(X_new)
    accuracy = accuracy_score(y_new, y_pred)
    
    print(f"✅ {adaptation_type} - Accuracy: {accuracy:.3f} - Time: {training_time:.2f}s")
    
    return {
        "model": model,
        "accuracy": accuracy,
        "adaptation_type": adaptation_type,
        "training_time": training_time
    }

# Demo Workflow - Run this to see SHARP in action
def run_demo():
    """Run the complete SHARP demo workflow"""
    print("=== SHARP Demo: Incremental Learning ===\n")
    
    # Generate synthetic continual learning tasks
    np.random.seed(42)
    
    # Task 0: Binary classification (class 0 vs 1)
    X0 = np.random.randn(200, 10)
    y0 = (X0[:, 0] + X0[:, 1] > 0).astype(int)
    
    # Task 1: Same classes but shifted distribution
    X1 = np.random.randn(200, 10) + 1.0  # Distribution shift
    y1 = (X1[:, 0] + X1[:, 1] > 0.5).astype(int)
    
    # Task 2: Another shift
    X2 = np.random.randn(200, 10) + 2.0
    y2 = (X2[:, 0] + X2[:, 1] > 1.0).astype(int)
    
    tasks = [(X0, y0), (X1, y1), (X2, y2)]
    
    # Run incremental learning
    previous_model = None
    results = []
    
    for task_id, (X, y) in enumerate(tasks):
        result = sharp_demo(previous_model, (X, y), task_id)
        previous_model = result["model"]
        #print(previous_model)
        results.append(result)
    
    # Summary
    print("\n=== Results Summary ===")
    for i, result in enumerate(results):
        print(f"Task {i}: {result['adaptation_type']} - "
              f"Accuracy: {result['accuracy']:.3f} - "
              f"Time: {result['training_time']:.2f}s")
    
    return results

# Run the demo
results = run_demo()

# Test the final model on new data
# print("\n=== Testing Final Model ===")
# final_model = results[-1]["model"]
# X_test = np.random.randn(50, 10) + 1.5
# y_test = (X_test[:, 0] + X_test[:, 1] > 0.75).astype(int)
#
# y_pred = final_model.predict(X_test)
# test_accuracy = accuracy_score(y_test, y_pred)
# print(f"Final model test accuracy: {test_accuracy:.3f}")
#
# print("\n🎉 SHARP Demo Complete!")