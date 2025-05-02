import os 
import numpy as np
import librosa
import tensorflow as tf
from tensorflow.keras.preprocessing.sequence import pad_sequences
from flask import Flask, request, jsonify, render_template
from flask_cors import CORS  # Add this import

app = Flask(__name__, static_folder='static', template_folder='templates')
CORS(app)  # Enable CORS for all routes

# Ensure uploads directory exists
if not os.path.exists('uploads'):
    os.makedirs('uploads')

# Load model only once when the app starts
model = tf.keras.models.load_model('models/deepfake_voice_model_new.h5')
max_length = 56239  # Adjust this based on your model's expected input length

def preprocess_audio(audio_path, max_length):
    try:
        audio, sr = librosa.load(audio_path, sr=None)
        mfccs = librosa.feature.mfcc(y=audio, sr=sr, n_mfcc=13)
        # Transpose mfccs to have shape (n_samples, n_features)
        mfccs = mfccs.T
        # Pad or truncate
        if len(mfccs) > max_length:
            padded_mfccs = mfccs[:max_length]
        else:
            # Create array of zeros with shape (max_length, n_features)
            padded_mfccs = np.zeros((max_length, mfccs.shape[1]))
            padded_mfccs[:mfccs.shape[0], :] = mfccs
            
        # Add batch dimension
        padded_mfccs = np.expand_dims(padded_mfccs, axis=0)
        return padded_mfccs
    except Exception as e:
        print(f"Error preprocessing audio: {str(e)}")
        raise

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    try:
        if 'audio' not in request.files:
            return jsonify({'error': 'No audio or voice file provided'}), 400

        audio = request.files['audio']
        
        if audio.filename == '':
            return jsonify({'error': 'No audio file selected'}), 400
            
        # Save the file
        audio_path = os.path.join('uploads', audio.filename)
        audio.save(audio_path)

        # Preprocess the audio file
        padded_sample = preprocess_audio(audio_path, max_length)

        # Make prediction
        prediction = model.predict(padded_sample)
        predicted_class = np.argmax(prediction, axis=1)[0]
        confidence = float(prediction[0][predicted_class])

        result = "Fake" if predicted_class == 0 else "Real"

        # Clean up
        if os.path.exists(audio_path):
            os.remove(audio_path)

        return jsonify({'prediction': result, 'confidence': confidence})
    except Exception as e:
        # Log the error
        print(f"Error processing request: {str(e)}")
        return jsonify({'error': f'Error processing audio: {str(e)}'}), 500

if __name__ == '__main__':
    app.run(debug=True, host='127.0.0.1', port=5000)