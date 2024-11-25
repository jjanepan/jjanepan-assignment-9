from flask import Flask, render_template, request, jsonify, send_from_directory
import os
from neural_networks import visualize
import uuid

app = Flask(__name__)

# Ensure the 'results' directory exists
RESULTS_DIR = "results"
os.makedirs(RESULTS_DIR, exist_ok=True)

@app.route('/')
def index():
    """
    Renders the main HTML page.
    """
    return render_template('index.html')

@app.route('/run_experiment', methods=['POST'])
def run_experiment():
    """
    Handles experiment parameters sent from the client and triggers the experiment.
    Returns the path to the generated result GIF.
    """
    try:
        # Extract parameters from the request
        data = request.json
        activation = data.get('activation')
        lr = float(data.get('lr', 0.01))  # Default learning rate
        step_num = int(data.get('step_num', 500))  # Default steps

        # Validate parameters
        if activation not in ['tanh', 'relu', 'sigmoid']:
            return jsonify({"error": "Invalid activation function. Choose from 'tanh', 'relu', or 'sigmoid'."}), 400
        
        if lr <= 0 or step_num <= 0:
            return jsonify({"error": "Learning rate and step number must be positive values."}), 400

        # Generate a unique filename for the result
        result_filename = f"visualize_{uuid.uuid4().hex}.gif"
        result_path = os.path.join(RESULTS_DIR, result_filename)

        # Run the experiment
        visualize(activation, lr, step_num, result_path)

        # Return the path to the result GIF
        return jsonify({"result_gif": f"/results/{result_filename}"})

    except Exception as e:
        return jsonify({"error": f"An error occurred: {str(e)}"}), 500

@app.route('/results/<filename>')
def results(filename):
    """
    Serves the generated result files.
    """
    return send_from_directory(RESULTS_DIR, filename)

if __name__ == '__main__':
    app.run(debug=True)
