import os
import librosa
import numpy as np
from sklearn.model_selection import train_test_split
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Conv1D, MaxPooling1D, Flatten, Dropout

# Function to extract Mel-frequency cepstral coefficients (MFCCs)
def extract_mfccs(audio_path, n_fft=2048, n_mfcc=20):
    y, sr = librosa.load(audio_path, sr=None)
    mfccs = np.mean(librosa.feature.mfcc(y=y, sr=sr, n_fft=n_fft, n_mfcc=n_mfcc).T, axis=0)
    return mfccs

# Function to load and preprocess voice data
def load_and_preprocess_data(data_path, classes, label_offset=0, n_mfcc=20):
    data = []
    labels = []
    for i, voice_class in enumerate(classes):
        class_path = os.path.join(data_path, voice_class)
        for filename in os.listdir(class_path):
            audio_path = os.path.join(class_path, filename)
            try:
                mfccs = extract_mfccs(audio_path, n_mfcc=n_mfcc)
                data.append(mfccs)
                labels.append(i + label_offset)
            except Exception as e:
                print(f"Error processing {audio_path}: {str(e)}")
    return np.array(data), np.array(labels)

# Define paths and classes for all voice data (including human and animals)
data_path = "C:\\Users\\vst04\\Desktop\\project\\mini proj(new)\\Data Train"
all_classes = ["cat", "chicken", "cow", "dog", "eagle", "human", "goat", "elephant"]

# Load and preprocess all voice data with 20 MFCC features
data, labels = load_and_preprocess_data(data_path, all_classes, n_mfcc=20)

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(data, labels, test_size=0.2, random_state=42)

# Reshape data for CNN input (channels=1 for single-channel audio)
X_train = X_train[:, :, np.newaxis]
X_test = X_test[:, :, np.newaxis]

# Model definition
model = Sequential()
model.add(Conv1D(32, kernel_size=3, activation='relu', input_shape=(X_train.shape[1], 1)))
model.add(MaxPooling1D(pool_size=2))
model.add(Flatten())
model.add(Dense(128, activation='relu'))
model.add(Dropout(0.2))
model.add(Dense(len(all_classes), activation='softmax'))  # Multiclass output

# Model compilation
model.compile(loss='sparse_categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

# Train the model
history = model.fit(X_train, y_train, epochs=20, validation_data=(X_test, y_test))

# Calculate overall accuracy
test_loss, test_accuracy = model.evaluate(X_test, y_test)
print(f"Overall Accuracy: {test_accuracy}")

# Save the model
model.save('animal_and_human_voice_recognition.h5')

# Predict function
def predict_audio(audio_file, n_mfcc=20):
    try:
        mfccs = extract_mfccs(audio_file, n_mfcc=n_mfcc)
        mfccs_reshaped = mfccs[np.newaxis, :, np.newaxis]
        prediction = np.argmax(model.predict(mfccs_reshaped), axis=1)
        if prediction[0] < len(all_classes):  # Check if predicted class index is within known classes
            return all_classes[prediction[0]]
        else:
            return "Unknown voice"
    except Exception as e:
        print(f"Error predicting audio: {str(e)}")
        return "Error"
