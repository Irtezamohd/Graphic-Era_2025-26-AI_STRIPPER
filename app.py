# import os
# import numpy as np
# import librosa
# import tensorflow as tf
# from flask import Flask, request, render_template, jsonify
# from tensorflow.keras.preprocessing.sequence import pad_sequences

# app = Flask(__name__)

# # Model load (Try-Except lagana behtar hai)
# try:
#     model = tf.keras.models.load_model('./model/lstm_model.h5')
# except Exception as e:
#     print(f"Error loading model: {e}")

# # Aapka max_length kaafi bada hai (56293), make sure aapki RAM handle kar sake
# max_length = 56293

# def preprocess_audio(audio_path, max_length):
#     # Audio load (16kHz standard)
#     audio, sr = librosa.load(audio_path, sr=16000) # sr fix karna behtar hota hai

#     # Feature extraction
#     mfcc = librosa.feature.mfcc(y=audio, sr=sr, n_mfcc=13)
#     delta = librosa.feature.delta(mfcc)
#     delta2 = librosa.feature.delta(mfcc, order=2)

#     # Combine (Shape: 39, Time) then Transpose to (Time, 39)
#     combined = np.vstack([mfcc, delta, delta2]).T

#     # Padding
#     padded = pad_sequences(
#         [combined],
#         maxlen=max_length,
#         dtype='float32',
#         padding='post',
#         truncating='post'
#     )
#     return padded

# @app.route('/')
# def home():
#     return render_template('index.html')

# @app.route('/predict', methods=['POST'])
# def predict():
#     if 'audio' not in request.files:
#         return jsonify({'error': 'No audio file provided'}), 400

#     audio_file = request.files['audio']
    
#     if audio_file.filename == '':
#         return jsonify({'error': 'No file selected'}), 400

#     # Variable name fix: audio_file use karein
#     audio_path = os.path.join("uploads", audio_file.filename)
#     audio_file.save(audio_path)

#     try:
#         # Preprocess
#         padded_sample = preprocess_audio(audio_path, max_length)

#         # Prediction
#         prediction = model.predict(padded_sample)
#         predicted_class = np.argmax(prediction, axis=1)[0]
#         confidence = float(np.max(prediction))
        
#         # Result Mapping
#         result = "Fake" if predicted_class == 1 else "Real"

#         # File use hone ke baad delete kar dena chahiye taaki server bhare nahi
#         os.remove(audio_path)

#         return jsonify({
#             'prediction': result,
#             'confidence': round(confidence, 4)
#         })
#     except Exception as e:
#         return jsonify({'error': str(e)}), 500

# if __name__ == '__main__':
#     if not os.path.exists('uploads'):
#         os.makedirs('uploads')
#     app.run(debug=True)


import os
import numpy as np
import librosa
import tensorflow as tf
from flask import Flask, request, render_template, jsonify
from tensorflow.keras.preprocessing.sequence import pad_sequences

app = Flask(__name__)

# Model loading logic
try:
    model = tf.keras.models.load_model(r'D:\AISTRIPPER\model\lstm_model.h5')
    print("✅ Model Loaded Successfully!")
except Exception as e:
    print(f"❌ Model Load Error: {e}")

max_length = 79

def preprocess_audio(audio_path, max_length):
    audio, sr = librosa.load(audio_path, sr=16000)
    mfcc = librosa.feature.mfcc(y=audio, sr=sr, n_mfcc=13)
    delta = librosa.feature.delta(mfcc)
    delta2 = librosa.feature.delta(mfcc, order=2)
    combined = np.vstack([mfcc, delta, delta2]).T
    padded = pad_sequences([combined], maxlen=max_length, dtype='float32', padding='post', truncating='post')
    return padded

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    try:
        if 'audio' not in request.files:
            return jsonify({'error': 'No audio file part'}), 400
        
        audio_file = request.files['audio']
        
        if audio_file.filename == '':
            return jsonify({'error': 'No selected file'}), 400

        # Create uploads folder if not exists
        if not os.path.exists('uploads'):
            os.makedirs('uploads')

        # Fix variable names here
        audio_path = os.path.join("uploads", audio_file.filename)
        audio_file.save(audio_path)

        # Process and Predict
        padded_sample = preprocess_audio(audio_path, max_length)
        prediction = model.predict(padded_sample)
        predicted_class = np.argmax(prediction, axis=1)[0]
        confidence = float(np.max(prediction))
        
        result = "Fake" if predicted_class == 1 else "Real"

        # Cleanup: Delete file after prediction
        if os.path.exists(audio_path):
            os.remove(audio_path)

        return jsonify({'prediction': result, 'confidence': round(confidence, 4)})

    except Exception as e:
        print(f"❌ Runtime Error: {str(e)}") # Ye error terminal mein dikhega
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    app.run(debug=True)