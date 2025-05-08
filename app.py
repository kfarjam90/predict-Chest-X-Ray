import os
import cv2
import numpy as np
from flask import Flask, render_template, request, redirect, url_for, flash
from werkzeug.utils import secure_filename
from tensorflow.keras.models import load_model
from tensorflow.keras.applications.resnet_v2 import preprocess_input
import logging

app = Flask(__name__)

# Configure logging
logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)

# Configuration
app.config['UPLOAD_FOLDER'] = 'uploads'
app.config['ALLOWED_EXTENSIONS'] = {'png', 'jpg', 'jpeg'}
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024  # 16MB limit
app.secret_key = 'your-secret-key-here'  # Needed for flash messages

try:
    # Load the pre-trained model with error handling
    model = load_model('model/resnet50v2_model.h5')
    logger.info("Model loaded successfully")
    #model.summary()  # This will print model architecture to verify
except Exception as e:
    logger.error(f"Error loading model: {str(e)}")
    raise e

class_names = ['COVID19', 'NORMAL', 'PNEUMONIA', 'TURBERCULOSIS']

def allowed_file(filename):
    return '.' in filename and \
           filename.rsplit('.', 1)[1].lower() in app.config['ALLOWED_EXTENSIONS']

def preprocess_image(image_path):
    try:
        # Read and preprocess the image
        logger.debug(f"Processing image: {image_path}")
        img = cv2.imread(image_path)
        if img is None:
            raise ValueError("Could not read image file")
            
        # Resize to 224x224 which is what ResNet50V2 expects by default
        img = cv2.resize(img, (224, 224))
        
        # Convert to RGB (OpenCV loads as BGR)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        
        # Expand dimensions and preprocess
        img = np.expand_dims(img, axis=0)
        img = preprocess_input(img)
        
        logger.debug(f"Image shape after preprocessing: {img.shape}")
        return img
    except Exception as e:
        logger.error(f"Error preprocessing image: {str(e)}")
        raise e

@app.route('/', methods=['GET'])
def index():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    if 'file' not in request.files:
        flash('No file selected')
        return redirect(request.url)
    
    file = request.files['file']
    if file.filename == '':
        flash('No file selected')
        return redirect(request.url)
    
    if file and allowed_file(file.filename):
        try:
            # Save the uploaded file
            filename = secure_filename(file.filename)
            os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)
            filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
            file.save(filepath)
            
            # Preprocess the image
            img = preprocess_image(filepath)
            
            # Make prediction
            logger.debug("Making prediction...")
            predictions = model.predict(img)
            logger.debug(f"Raw predictions: {predictions}")
            
            predicted_class = class_names[np.argmax(predictions)]
            confidence = round(100 * np.max(predictions), 2)
            
            # Remove the uploaded file after processing
            os.remove(filepath)
            
            return render_template('result.html', 
                                 image_name=filename,
                                 prediction=predicted_class,
                                 confidence=confidence,
                                 class_names=class_names,
                                 probabilities=[round(p*100, 2) for p in predictions[0]])
        
        except Exception as e:
            logger.error(f"Error during prediction: {str(e)}")
            flash(f'Error processing image: {str(e)}')
            return redirect(request.url)
    
    flash('Allowed file types are png, jpg, jpeg')
    return redirect(request.url)

if __name__ == '__main__':
    os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)
    app.run(debug=True)