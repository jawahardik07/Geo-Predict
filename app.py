from flask import Flask, render_template, request
from joblib import load
import numpy as np
import logging
import os

# --- Configuration ---
app = Flask(__name__)
# Set this to ANY string. For example:
app.config['SECRET_KEY'] = 'my_temporary_secret_key_123!' # Or generate a random one

# --- Logging Setup ---
if not os.path.exists('logs'):
    os.makedirs('logs')
log_file_path = os.path.join('logs', 'geo_predict_app.log')
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(log_file_path),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

# --- Load Models and Scaler ---
try:
    model = load('geo_predict_model.pkl')
    scaler = load('geo_predict_scaler.pkl')
    label_encoder = load('geo_predict_label_encoder.pkl')
    logger.info("Model, scaler, and label encoder loaded successfully.")
except FileNotFoundError as e:
    logger.error(f"Error loading .pkl files: {e}. Ensure they are in the root directory with app.py (e.g., C:\\Hj major project\\).")
    model = None
    scaler = None
    label_encoder = None
except Exception as e:
    logger.error(f"An unexpected error occurred while loading .pkl files: {e}")
    model = None
    scaler = None
    label_encoder = None

# --- Route ---
@app.route('/', methods=['GET', 'POST'])
def index():
    prediction_result = None
    error_message = None
    input_values_repopulate = {} # To repopulate form on error or reload

    if request.method == 'POST':
        if not all([model, scaler, label_encoder]):
            logger.error("Model/scaler/encoder not loaded. Cannot predict.")
            error_message = "CRITICAL ERROR: Prediction models could not be loaded. Please check server logs."
            # Capture current inputs to repopulate form
            for key in ['sio2', 'al2o3', 'feo', 'mgo', 'cao', 'na2o', 'k2o', 'tio2']:
                input_values_repopulate[key] = request.form.get(key, '')
            return render_template('index.html', prediction_result=None, error_message=error_message, input_values=input_values_repopulate)

        try:
            # Get data directly from the form submission
            sio2_val = request.form.get('sio2')
            al2o3_val = request.form.get('al2o3')
            feo_val = request.form.get('feo')
            mgo_val = request.form.get('mgo')
            cao_val = request.form.get('cao')
            na2o_val = request.form.get('na2o')
            k2o_val = request.form.get('k2o')
            tio2_val = request.form.get('tio2')

            # Store for repopulating form in case of error after this point
            input_values_repopulate = {
                'sio2': sio2_val, 'al2o3': al2o3_val, 'feo': feo_val, 'mgo': mgo_val,
                'cao': cao_val, 'na2o': na2o_val, 'k2o': k2o_val, 'tio2': tio2_val
            }

            required_form_fields = [sio2_val, al2o3_val, feo_val, mgo_val, cao_val, na2o_val, k2o_val, tio2_val]
            if not all(required_form_fields):
                error_message = "All chemical composition fields are required. Please enter a value for each."
            else:
                try:
                    data_values_float = [
                        float(sio2_val), float(al2o3_val), float(feo_val), float(mgo_val),
                        float(cao_val), float(na2o_val), float(k2o_val), float(tio2_val)
                    ]
                    
                    if not all(0 <= val <= 100 for val in data_values_float):
                        error_message = "All input values must be numbers between 0 and 100."
                    else:
                        data_array = np.array([data_values_float])
                        scaled_data = scaler.transform(data_array)
                        prediction_encoded = model.predict(scaled_data)[0]
                        prediction_result = label_encoder.inverse_transform([prediction_encoded])[0]
                        logger.info(f"Prediction: {prediction_result} for input: {input_values_repopulate}")
                        # Clear error message if prediction is successful
                        error_message = None 

                except ValueError:
                    error_message = "Invalid input: All chemical compositions must be valid numbers (e.g., 12.34)."
                except Exception as e: # Catch other prediction-related errors
                    logger.error(f"Error during prediction processing: {e}", exc_info=True)
                    error_message = f"An error occurred during prediction: {e}"
        
        except Exception as e: # Catch broader form processing errors
            logger.error(f"Error processing form data: {e}", exc_info=True)
            error_message = "An critical error occurred processing your request. Check logs."

    # For GET requests, or after POST if prediction was made (or error occurred)
    return render_template('index.html', prediction_result=prediction_result, error_message=error_message, input_values=input_values_repopulate)

# --- Basic Error Handling Pages (Optional but good) ---
@app.errorhandler(404)
def page_not_found(e):
    logger.warning(f"404 error: Page not found at {request.url} - {e}")
    return "<h1>404 - Page Not Found</h1><p>The page you are looking for does not exist.</p><a href='/'>Go Home</a>", 404

@app.errorhandler(500)
def internal_server_error(e):
    logger.error(f"500 Internal Server Error: {e} at {request.url}", exc_info=True)
    return "<h1>500 - Internal Server Error</h1><p>Something went wrong on our side. Please try again later.</p><a href='/'>Go Home</a>", 500

# --- Main Execution ---
if __name__ == '__main__':
    port = int(os.environ.get("PORT", 5000))
    # Set debug=True ONLY for local testing if you need more detailed browser errors.
    # For submission, debug=False is safer if it's running.
    app.run(host='0.0.0.0', port=port, debug=False)