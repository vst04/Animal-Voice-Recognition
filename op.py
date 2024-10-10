import os
import librosa
import numpy as np
from sklearn.model_selection import train_test_split
from tensorflow.keras.models import load_model
from tensorflow.keras.utils import to_categorical

# Function to extract Mel-frequency cepstral coefficients (MFCCs)
def extract_mfccs(audio_path, n_fft=2048):  # Adjust n_fft if needed
    y, sr = librosa.load(audio_path)
    mfccs = np.mean(librosa.feature.mfcc(y=y, sr=sr, n_fft=n_fft).T, axis=0)
    return mfccs

# Load the trained model
model = load_model('animal_and_human_voice_recognition.h5')

# Function to make predictions on an audio file
def predict_animal(audio_path):
    # Preprocess the audio file
    mfccs = extract_mfccs(audio_path)
    data = mfccs[np.newaxis, :, np.newaxis]  # Reshape for CNN input
    
    # Perform prediction using the loaded model
    prediction = model.predict(data)
    
    # Get the predicted class label
    predicted_class = np.argmax(prediction)
    classes = ["cat", "chicken", "cow", "dog", "eagle", "human", "goat", "elephant"]
    predicted_label = classes[predicted_class]
    
    return predicted_label

# Example usage
audio_file_path = "C:\\Users\\vst04\\Desktop\\project\\mini proj(new)\\Data Train\\elephant\\Sound 03.wav"
predicted_animal = predict_animal(audio_file_path)
print("Predicted Animal:", predicted_animal)
