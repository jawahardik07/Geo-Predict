# Geo-Predict
Predicts mineral names based on their chemical makeup using a machine learning model and a Flask web interface.
# GeoPredict - Mineral Analysis Platform

## Description
This project is a web application built with Python and Flask that predicts mineral names based on their chemical compositions using a pre-trained machine learning model.
(Add another sentence or two about your project if you like).

## Features
* Accepts chemical composition inputs (SiO2, Al2O3, FeO, MgO, CaO, Na2O, K2O, TiO2).
* Predicts the mineral name.
* User-friendly web interface.
(List any other features you implemented in the version you uploaded).

## Setup and Running the Application Locally

1.  **Clone or Download:**
    * Download the project files from this GitHub repository (e.g., using the "Code" button -> "Download ZIP").
    * Extract the files to a folder on your computer.

2.  **Large Model Files (If Applicable):**
    * *(Only include this section if you had to host large .pkl files on Google Drive/Dropbox, etc.)*
    * The main model file (`your_large_model_filename.pkl`) is hosted separately due to its size.
    * Please download it from: [Your Shareable Link to the Large Model File Here]
    * Place this downloaded `.pkl` file into the main project directory (the same folder as `app.py`).
    * *(Add similar instructions if your `expanded_dataset.csv` was also hosted separately).*

3.  **Create and Activate a Virtual Environment:**
    * Open your terminal or PowerShell.
    * Navigate to the project directory: `cd path/to/your/GeoPredict-Project`
    * Create a virtual environment: `python -m venv venv`
    * Activate it:
        * Windows: `.\venv\Scripts\activate`
        * macOS/Linux: `source venv/bin/activate`

4.  **Install Dependencies:**
    * With the virtual environment active, install the required Python packages:
        ```bash
        pip install -r requirements.txt
        ```

5.  **Run the Application:**
    * In the same terminal (with the virtual environment still active), run:
        ```bash
        python app.py
        ```

6.  **Access in Browser:**
    * Open your web browser and go to: `http://127.0.0.1:5000/`

## Project Structure
* `app.py`: The main Flask application file.
* `templates/`: Contains the HTML file (`index.html`).
* `static/`: Contains the CSS file (`styles.css`).
* `*.pkl`: Pre-trained machine learning model, scaler, and encoder files.
* `requirements.txt`: List of Python dependencies.
* `.gitignore`: Specifies intentionally untracked files that Git should ignore.
* `logs/`: Contains log files (this folder will be created when the app runs).

(Add any other important notes for your teacher here).
