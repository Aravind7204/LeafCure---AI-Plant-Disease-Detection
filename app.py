from flask import Flask, request, render_template, jsonify, send_from_directory
import tensorflow as tf
import numpy as np
import os
from werkzeug.utils import secure_filename
import json

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = 'uploads'
ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg', 'gif'}

# Load the trained model globally
model = tf.keras.models.load_model('trained_plant_disease_model.keras')

with open('./static/solutions/solutions.json', 'r') as file:
    solutions = json.load(file)

# Function to check allowed file types
def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

# Function to make predictions
def model_prediction(image_path):
    try:
        image = tf.keras.preprocessing.image.load_img(image_path, target_size=(64, 64))  # Match input size for your model
        input_arr = tf.keras.preprocessing.image.img_to_array(image)
        input_arr = np.array([input_arr])  # Convert single image to batch
        predictions = model.predict(input_arr)
        
        # Get the predicted class and the associated probability
        predicted_class = np.argmax(predictions)  # Class with the highest probability
        predicted_probability = np.max(predictions)  # The highest probability value
        
        return predicted_class, predicted_probability
    except Exception as e:
        print(f"Error in prediction: {e}")
        raise

# Route to serve uploaded files
@app.route('/uploads/<filename>')
def uploaded_file(filename):
    return send_from_directory(app.config['UPLOAD_FOLDER'], filename)

# Routes
@app.route('/')
def home():
    return render_template('index.html')


@app.route('/terms')
def terms():
    return render_template('terms.html')

@app.route('/privacy')
def privacy():
    return render_template('privacy-policy.html')

@app.route('/about')  # Route for the About Developer page
def about():
    return render_template('about.html')

@app.route('/detect')
def detect_page():
    return render_template('aipage.html')

@app.route('/predict', methods=['POST'])
def predict():
    if 'file' not in request.files:
        return jsonify({"error": "No file uploaded"}), 400

    file = request.files['file']
    if file.filename == '':
        return jsonify({"error": "No file selected"}), 400

    if not allowed_file(file.filename):
        return jsonify({"error": "Invalid file type. Only image files are allowed."}), 400

    filename = secure_filename(file.filename)
    file_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
    try:
        file.save(file_path)
        
        # Make prediction
        result_index, probability = model_prediction(file_path)
        
        # Class labels
        class_name = ['Apple___Apple_scab', 'Apple___Black_rot', 'Apple___Cedar_apple_rust', 'Apple___healthy',
                      'Blueberry___healthy', 'Cherry_(including_sour)___Powdery_mildew', 
                      'Cherry_(including_sour)___healthy', 'Corn_(maize)___Cercospora_leaf_spot Gray_leaf_spot', 
                      'Corn_(maize)___Common_rust_', 'Corn_(maize)___Northern_Leaf_Blight', 'Corn_(maize)___healthy', 
                      'Grape___Black_rot', 'Grape___Esca_(Black_Measles)', 'Grape___Leaf_blight_(Isariopsis_Leaf_Spot)', 
                      'Grape___healthy', 'Orange___Haunglongbing_(Citrus_greening)', 'Peach___Bacterial_spot',
                      'Peach___healthy', 'Pepper,_bell___Bacterial_spot', 'Pepper,_bell___healthy', 
                      'Potato___Early_blight', 'Potato___Late_blight', 'Potato___healthy', 
                      'Raspberry___healthy', 'Soybean___healthy', 'Squash___Powdery_mildew', 
                      'Strawberry___Leaf_scorch', 'Strawberry___healthy', 'Tomato___Bacterial_spot', 
                      'Tomato___Early_blight', 'Tomato___Late_blight', 'Tomato___Leaf_Mold', 
                      'Tomato___Septoria_leaf_spot', 'Tomato___Spider_mites Two-spotted_spider_mite', 
                      'Tomato___Target_Spot', 'Tomato___Tomato_Yellow_Leaf_Curl_Virus', 'Tomato___Tomato_mosaic_virus',
                      'Tomato___healthy']

        predicted_label = class_name[result_index]
        return jsonify({
            "label": predicted_label,
            "probability": float(probability),  # Convert the probability to a float for JSON serialization
            "filename": filename,
            "file_url": f"/uploads/{filename}"  # Provide a relative path to the uploaded file
        })
    
    except Exception as e:
        return jsonify({"error": f"Unable to process the image: {str(e)}"}), 500
    
@app.route('/get_solution', methods=['POST'])
def get_solution():
    disease_name = request.json.get('disease_name')  # Ensure the correct key name
    solution = solutions.get(disease_name, "Solution not found for this disease.")
    return jsonify({"disease_name": disease_name, "solution": solution})



if __name__ == '__main__':
    os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)
    app.run(debug=False,host='0.0.0.0')
