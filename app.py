from flask import Flask, request, jsonify, render_template
from flask_cors import CORS
import tensorflow as tf
import numpy as np
from PIL import Image
import io

# Initialize the Flask app and CORS
app = Flask(__name__)
CORS(app)  # Enable Cross-Origin Resource Sharing (CORS)

# Load your trained model
model = tf.keras.models.load_model('path_to_your_model.h5')  # Replace with your model path
solutions = {
    "Apple___Apple_scab": {
        "solution": "Apply fungicide to control the disease.",
        "product_link": "https://example.com/fungicide"
    },
    "Apple___healthy": {
        "solution": "No disease detected. Keep monitoring."
    }
}

# Route for the main page
@app.route('/')
def index():
    return render_template('index.html')

# Route to handle image upload and prediction
@app.route('/predict', methods=['POST'])
def predict():
    if 'file' not in request.files:
        return jsonify({'error': 'No file part'}), 400
    
    file = request.files['file']
    
    if file.filename == '':
        return jsonify({'error': 'No selected file'}), 400

    try:
        # Load and process image
        image = Image.open(file.stream)
        image = image.resize((224, 224))  # Adjust size according to your model
        image = np.array(image) / 255.0  # Normalize image if necessary
        image = np.expand_dims(image, axis=0)  # Add batch dimension

        # Predict using the model
        prediction = model.predict(image)
        predicted_class = np.argmax(prediction, axis=1)
        predicted_label = str(predicted_class[0])  # This should match your label
        predicted_probability = np.max(prediction)

        return jsonify({
            'label': predicted_label,
            'probability': predicted_probability
        })

    except Exception as e:
        return jsonify({'error': str(e)}), 500

# Route to get the solution based on disease name
@app.route('/get_solution', methods=['POST'])
def get_solution():
    data = request.get_json()
    disease_name = data.get('disease_name')

    # Retrieve the solution for the disease
    solution = solutions.get(disease_name)
    
    if solution:
        return jsonify({"solution": solution})
    else:
        return jsonify({"solution": "No solution found for this disease."})

if __name__ == '__main__':
    app.run(debug=True)
