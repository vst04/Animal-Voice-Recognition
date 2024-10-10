from flask import Flask, request, jsonify, render_template
from tensorflow.keras.models import load_model
import numpy as np
import librosa
import soundfile as sf
import io

app = Flask(__name__)

# Specify the full path to the HDF5 file
model_path = 'C:\\Users\\vst04\\Desktop\\project\\mini proj(new)\\animal_and_human_voice_recognition.h5'

# Load the trained model
model = load_model(model_path)

# Define a function to preprocess the audio file
def preprocess_audio(file):
    # Load the audio file using librosa
    y, sr = librosa.load(io.BytesIO(file.read()), sr=None)
    mfccs = np.mean(librosa.feature.mfcc(y=y, sr=sr, n_fft=2048).T, axis=0)
    return mfccs

# Define a function to make predictions
def predict_animal(audio_file):
    # Preprocess the audio file
    preprocessed_audio = preprocess_audio(audio_file)
    
    # Perform prediction using the loaded model
    prediction_probabilities = model.predict(preprocessed_audio.reshape(1, -1, 1))[0]
    
    # Define animal classes
    animal_classes = ['cat', 'chicken', 'cow', 'dog', 'eagle', 'human', 'goat', 'elephant']
    
    # Decode the prediction into human-readable format
    predicted_class_index = np.argmax(prediction_probabilities)
    predicted_class = animal_classes[predicted_class_index]
    
    return predicted_class

@app.route('/')
def index():
    return render_template('getstarted.html')

@app.route('/index')
def index_page():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    if 'audio' not in request.files:
        return jsonify({'error': 'No audio file provided'}), 400
    
    audio_file = request.files['audio']
    
    # Make prediction
    prediction = predict_animal(audio_file)
    
    # Return prediction result
    return jsonify({'prediction': prediction})

# Define routes for individual animals
@app.route('/cat')
def cat_page():
    return render_template('cat.html')

@app.route('/chicken')
def chicken_page():
    return render_template('chicken.html')

@app.route('/cow')
def cow_page():
    return render_template('cow.html')

@app.route('/dog')
def dog_page():
    return render_template('dog.html')

@app.route('/eagle')
def eagle_page():
    return render_template('eagle.html')

@app.route('/human')
def human_page():
    return render_template('human.html')

@app.route('/goat')
def goat_page():
    return render_template('goat.html')

@app.route('/elephant')
def elephant_page():
    return render_template('elephant.html')

# Route for unknown prediction
@app.route('/unknown')
def unknown_page():
    return render_template('unknown.html')

if __name__ == '__main__':
    app.run(debug=True)
