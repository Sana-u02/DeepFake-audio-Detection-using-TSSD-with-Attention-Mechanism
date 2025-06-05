

# import libraries  
from flask import Flask, render_template, request, session, g
import numpy as np
import sqlite3
import re
import pandas as pd
import librosa
import os
import time

# Create a Flask application instance
app = Flask(__name__)
# Set a secret key for the session
app.secret_key = "KjhLJF54f6ds234H"

# Define the database name
DATABASE = "mydb.sqlite3"


# Load the model 
model_namee = "tssd_attention.pth" 
#model_name = pd.read_csv('tssd_model.h5.csv')

# Set parameters for audio feature extraction
num_mfcc = 100
num_mels = 128
num_chroma = 50

# Function to get a database connection
def get_db():
    # Check if there is an existing database connection
    db = getattr(g, "_database", None)
    if db is None:
        # If not, create a new database connection
        db = g._database = sqlite3.connect(DATABASE)
    return db

# Function to close the database connection after the request context ends
@app.teardown_appcontext
def close_connection(exception):
    db = getattr(g, "_database", None)
    if db is not None:
        db.close()

# Route for the home page
@app.route('/')
def home():
    # Set the background image for the home page
    background_image = "/static/image1.jpg"
    return render_template('index.html', background_image=background_image)

# Route for the login page
@app.route('/login.html', methods=['GET', 'POST'])
def login():
    background_image = "/static/image5.jpg"
    if request.method == 'POST':
        # Get email and password from the form
        email = request.form["email"]
        password = request.form["password"]
        cursor = get_db().cursor()
        # Check if the user exists in the database
        cursor.execute("SELECT * FROM REGISTER WHERE EMAIL = ? AND PASSWORD = ?", (email, password))
        account = cursor.fetchone()
        if account:
            # If user is found, set session variables
            session['Loggedin'] = True
            session['id'] = account[1]
            session['email'] = account[1]
            return render_template('model.html', background_image=background_image)
        else:
            msg = "Incorrect Email/password"
            return render_template('login.html', msg=msg, background_image=background_image)
    else:
        # If GET request, render login page
        return render_template('login.html', background_image=background_image)

# Route for the contact page
@app.route('/contact.html')
def contact():
    background_image = "/static/image3.jpg"
    return render_template('contact.html', background_image=background_image)

# Route for the about page
@app.route('/about.html')
def about():
    background_image = "/static/image2.jpg"
    return render_template('about.html', background_image=background_image)

# Redirect route for index.html (home page)
@app.route('/index.html')
def home1():
    background_image = "/static/image1.jpg"
    return render_template('index.html', background_image=background_image)

# Route for the signup page
@app.route('/register.html', methods=['GET', 'POST'])
def signup():
    msg = ''
    background_image = "/static/image4.jpg"
    if request.method == 'POST':
        # Get registration details from the form
        username = request.form["username"]
        email = request.form["email"]
        password = request.form["password"]
        confirm_password = request.form["confirm-password"]
        cursor = get_db().cursor()
        # Check for existing username and email
        cursor.execute("SELECT * FROM REGISTER WHERE username = ?", (username,))
        account_username = cursor.fetchone()
        cursor.execute("SELECT * FROM REGISTER WHERE email = ?", (email,))
        account_email = cursor.fetchone()

        # Validate input fields
        if account_username:
            msg = "Username already exists"
        elif account_email:
            msg = "Email already exists"
        elif not re.match(r'[^@]+@[^@]+\.[^@]+', email):
            msg = "Invalid Email Address!"
        elif password != confirm_password:
            msg = "Passwords do not match!"
        else:
            # Insert new user into the database if validation passes
            cursor.execute("INSERT INTO REGISTER (username, email, password) VALUES (?,?,?)",
                           (username, email, password))
            get_db().commit()
            msg = "You have successfully registered"

    return render_template('register.html', msg=msg, background_image=background_image)

# Route for the model prediction page
@app.route('/model.html', methods=['GET', 'POST'])
def model():
    background_image = "/static/image5.jpg"
    loader_visible = False

    if request.method == 'POST':
        # Get the uploaded audio file
        selected_file = request.files['audio_file']
        file_name = selected_file.filename

        file_stream = selected_file.stream
        file_stream.seek(0)

        # Load audio and extract features using librosa
        X, sample_rate = librosa.load(file_stream, res_type='kaiser_fast')
        mfccs = np.mean(librosa.feature.mfcc(y=X, sr=sample_rate, n_mfcc=num_mfcc).T, axis=0)
        mel_spectrogram = np.mean(librosa.feature.melspectrogram(y=X, sr=sample_rate, n_mels=num_mels).T, axis=0)
        chroma_features = np.mean(librosa.feature.chroma_stft(y=X, sr=sample_rate, n_chroma=num_chroma).T, axis=0)
        zcr = np.mean(librosa.feature.zero_crossing_rate(y=X).T, axis=0)
        spectral_centroid = np.mean(librosa.feature.spectral_centroid(y=X, sr=sample_rate).T, axis=0)
        flatness = np.mean(librosa.feature.spectral_flatness(y=X).T, axis=0)
        
        # Concatenate all features into one array
        features = np.concatenate((mfccs, mel_spectrogram, chroma_features, zcr, spectral_centroid, flatness))
        
        # Compare the model to find the closest match
        
        distances = np.linalg.norm(model_name.iloc[:, :-1] - features, axis=1)
        closest_match_idx = np.argmin(distances)
        closest_match_label = model_name.iloc[closest_match_idx, -1]
        total_distance = np.sum(distances)
        closest_match_prob = 1 - (distances[closest_match_idx] / total_distance)
        closest_match_prob_percentage = "{:.3f}".format(closest_match_prob * 100)

        # Prepare result labels based on the closest match
        if closest_match_label == 'deepfake':
            file_label = f"File: {file_name}"
            result_label = f"Result: Fake with {closest_match_prob_percentage}%"
        else:
            file_label = f"File: {file_name}"
            result_label = f"Result: Real with {closest_match_prob_percentage}%"

        return render_template('model.html', file_label=file_label, result_label=result_label, background_image=background_image, loader_visible=loader_visible)
    else:
        # If GET request, render the model page
        return render_template('model.html', background_image=background_image, loader_visible=loader_visible)

# Run the Flask application in debug mode
if __name__ == "__main__":
    app.run(debug=True)

