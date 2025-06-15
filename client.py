from flask import Flask, render_template, request, jsonify
import requests
import os
import logging

# Configure logging
logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)

app = Flask(__name__)

@app.route('/')
def index():
    logger.debug("Rendering index page")
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    logger.debug("Sending file to server for prediction")
    
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
            # Send file to server
            server_url = 'http://127.0.0.1:5000/upload'
            files = {'file': (file.filename, file.stream, file.content_type)}
            data = {'model_type': model_type}
            logger.debug("Sending request to server...")
            
            response = requests.post(server_url, files=files, data=data)
            response.raise_for_status()
            
            # Parse server response
            predictions = response.json()
            logger.debug(f"Received predictions: {predictions}")
            
            return jsonify(predictions)
            
        except Exception as e:
            logger.error(f"Error sending file to server: {str(e)}", exc_info=True)
            return jsonify({'error': str(e)}), 500
            
    logger.error("Invalid file type")
    return jsonify({'error': 'Invalid file type'}), 400

if __name__ == '__main__':
    app.run(debug=True, port=5001)