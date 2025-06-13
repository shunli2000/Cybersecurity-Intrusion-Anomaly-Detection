from flask import Flask, render_template, request, jsonify, send_from_directory
import os
import pandas as pd
from utils import process_input_data, predict_with_model
import logging

# Configure logging
logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)

app = Flask(__name__, 
            static_folder='static',
            template_folder='templates')

# Configure upload folder
UPLOAD_FOLDER = 'uploads'
if not os.path.exists(UPLOAD_FOLDER):
    os.makedirs(UPLOAD_FOLDER)
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

@app.route('/')
def index():
    logger.debug("Rendering index page")
    return render_template('index.html')

@app.route('/static/<path:filename>')
def serve_static(filename):
    logger.debug(f"Serving static file: {filename}")
    return send_from_directory(app.static_folder, filename)

@app.route('/upload', methods=['POST'])
def upload_file():
    logger.debug("Received file upload request")
    
    if 'file' not in request.files:
        logger.error("No file part in request")
        return jsonify({'error': 'No file part'}), 400
    
    file = request.files['file']
    model_type = request.form.get('model_type', 'ifor')
    logger.debug(f"Model type selected: {model_type}")
    
    if file.filename == '':
        logger.error("No selected file")
        return jsonify({'error': 'No selected file'}), 400
    
    if file and file.filename.endswith('.csv'):
        try:
            # Save the uploaded file
            filepath = os.path.join(app.config['UPLOAD_FOLDER'], file.filename)
            file.save(filepath)
            logger.debug(f"File saved to: {filepath}")
            
            # Process the input data
            logger.debug("Processing input data")
            processed_data = process_input_data(filepath)
            
            # Get predictions
            logger.debug("Getting predictions")
            predictions = predict_with_model(processed_data, model_type)
            
            # Convert predictions to a format suitable for display
            results = []
            for i, pred in enumerate(predictions):
                results.append({
                    'index': i,
                    'prediction': int(pred)  # Convert to int for JSON serialization
                })
            
            logger.debug(f"Generated {len(results)} predictions")
            return jsonify({
                'success': True,
                'results': results
            })
            
        except Exception as e:
            logger.error(f"Error processing file: {str(e)}", exc_info=True)
            return jsonify({'error': str(e)}), 500
            
    logger.error("Invalid file type")
    return jsonify({'error': 'Invalid file type'}), 400

if __name__ == '__main__':
    app.run(debug=True, port=5000) 