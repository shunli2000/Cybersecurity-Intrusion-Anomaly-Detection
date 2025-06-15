from flask import Flask, request, jsonify
import os
from utils import process_input_data, predict_with_model
import logging

# Configure logging
logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)

app = Flask(__name__)

# Configure upload folder
UPLOAD_FOLDER = 'uploads'
if not os.path.exists(UPLOAD_FOLDER):
    os.makedirs(UPLOAD_FOLDER)
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

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
            # Expecting process_input_data to return scaled_data and original_info_df
            processed_data, original_info_df = process_input_data(filepath)
            
            # Get predictions
            logger.debug("Getting predictions")
            predictions = predict_with_model(processed_data, model_type)
            
            # Convert predictions to a format suitable for display
            # and include timestamp and userId
            results = []
            anomaly_count = 0
            total_predictions = len(predictions)

            for i, pred in enumerate(predictions):
                results.append({
                    'timestamp': original_info_df.iloc[i]['timestamp'],
                    'userId': original_info_df.iloc[i]['userId'],
                    'prediction': int(pred)
                })
                if int(pred) == 1:
                    anomaly_count += 1
            
            anomaly_rate = 0
            if total_predictions > 0:
                anomaly_rate = (anomaly_count / total_predictions) * 100
            
            logger.debug(f"Generated {len(results)} predictions. Anomaly rate: {anomaly_rate:.2f}%")
            # print(results) # For server-side debugging
            return jsonify({
                'success': True,
                'results': results,
                'anomaly_rate': float(f"{anomaly_rate:.2f}") # Format to 2 decimal places
            })
            
        except Exception as e:
            logger.error(f"Error processing file: {str(e)}", exc_info=True)
            return jsonify({'error': str(e)}), 500
            
    logger.error("Invalid file type")
    return jsonify({'error': 'Invalid file type'}), 400

if __name__ == '__main__':
    app.run(debug=True, port=5000)