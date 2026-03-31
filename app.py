


# import os
# import numpy as np
# import librosa
# import tensorflow as tf
# from flask import Flask, request, render_template, jsonify
# from tensorflow.keras.preprocessing.sequence import pad_sequences

# app = Flask(__name__)

# # Model loading logic
# try:
#     model = tf.keras.models.load_model('D:/AISTRIPPER/Graphic-Era_2025-26-AI_STRIPPER/notebook/.ipynb_checkpoints/lstm_model.h5')
#     print("✅ Model Loaded Successfully!")
# except Exception as e:
#     print(f"❌ Model Load Error: {e}")

# max_length = 79

# def preprocess_audio(audio_path, max_length):
#     audio, sr = librosa.load(audio_path, sr=16000)
#     mfcc = librosa.feature.mfcc(y=audio, sr=sr, n_mfcc=13)
#     mfcc.T
#     padded = pad_sequences([mfcc], maxlen=max_length, dtype='float32', padding='post', truncating='post')
#     return padded

# @app.route('/')
# def home():
#     return render_template('index.html')

# @app.route('/predict', methods=['POST'])
# def predict():
#     try:
#         if 'audio' not in request.files:
#             return jsonify({'error': 'No audio file part'}), 400
        
#         audio_file = request.files['audio']
        
#         if audio_file.filename == '':
#             return jsonify({'error': 'No selected file'}), 400

#         # Create uploads folder if not exists
#         if not os.path.exists('uploads'):
#             os.makedirs('uploads')

#         # Fix variable names here
#         audio_path = os.path.join("uploads", audio_file.filename)
#         audio_file.save(audio_path)

#         # Process and Predict
#         padded_sample = preprocess_audio(audio_path, max_length)
#         prediction = model.predict(padded_sample)
#         predicted_class = np.argmax(prediction, axis=1)[0]
#         confidence = float(np.max(prediction))
        
#         result = "Fake" if predicted_class == 1 else "Real"

#         # Cleanup: Delete file after prediction
#         if os.path.exists(audio_path):
#             os.remove(audio_path)

#         return jsonify({'prediction': result, 'confidence': round(confidence, 4)})

#     except Exception as e:
#         print(f"❌ Runtime Error: {str(e)}") # Ye error terminal mein dikhega
#         return jsonify({'error': str(e)}), 500

# if __name__ == '__main__':
#     app.run(debug=True)


import os
import numpy as np
import librosa
import tensorflow as tf
from flask import Flask, request, render_template, jsonify
from tensorflow.keras.preprocessing.sequence import pad_sequences

# Reduce TF logs
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

app = Flask(__name__)

# -------------------------------
# LOAD MODEL
# -------------------------------
try:
    print("🚀 Loading model...")
    model = tf.keras.models.load_model(
        r'D:\AISTRIPPER\Graphic-Era_2025-26-AI_STRIPPER\notebook\.ipynb_checkpoints\lstm_model.h5'
    )
    print("✅ Model Loaded Successfully!")

except Exception as e:
    print(f"❌ Model Load Error: {e}")
    model = None

# Auto detect max_length
max_length = model.input_shape[1]
print("📏 Max Length:", max_length)

# -------------------------------
# PREPROCESS (13 MFCC ONLY)
# -------------------------------
def preprocess_audio(audio_path, max_length):
    try:
        audio, sr = librosa.load(audio_path, sr=16000)

        # 13 MFCC ONLY
        mfcc = librosa.feature.mfcc(y=audio, sr=sr, n_mfcc=13).T

        padded = pad_sequences(
            [mfcc],
            maxlen=max_length,
            dtype='float32',
            padding='post',
            truncating='post'
        )

        return padded

    except Exception as e:
        print("❌ Preprocess error:", e)
        return None


# -------------------------------
# ROUTES
# -------------------------------
@app.route('/')
def home():
    return render_template('index.html')


@app.route('/predict', methods=['POST'])
def predict():
    try:
        print("\n🔥 Request received")

        if model is None:
            return jsonify({'error': 'Model not loaded'}), 500

        if 'audio' not in request.files:
            return jsonify({'error': 'No file uploaded'}), 400

        file = request.files['audio']

        if file.filename == '':
            return jsonify({'error': 'Empty filename'}), 400

        # Save file
        os.makedirs('uploads', exist_ok=True)
        file_path = os.path.join('uploads', file.filename)
        file.save(file_path)

        print("📂 File saved:", file_path)

        # Preprocess
        padded_sample = preprocess_audio(file_path, max_length)

        if padded_sample is None:
            return jsonify({'error': 'Preprocessing failed'}), 500

        print("📊 Shape:", padded_sample.shape)

        # Prediction
        prediction = model.predict(padded_sample, verbose=0)

        predicted_class = int(np.argmax(prediction))
        confidence = float(np.max(prediction))

        result = "Fake" if predicted_class == 1 else "Real"

        print("✅ Result:", result, confidence)

        # Delete file after use
        os.remove(file_path)

        return jsonify({
            'prediction': result,
            'confidence': round(confidence, 4)
        })

    except Exception as e:
        print("❌ ERROR:", str(e))
        return jsonify({'error': str(e)}), 500


# -------------------------------
# RUN SERVER
# -------------------------------
if __name__ == '__main__':
    app.run(debug=True, use_reloader=False)