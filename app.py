from flask import Flask, render_template, request, redirect, jsonify
import numpy as np
import pandas as pd
import tensorflow as tf
import warnings
warnings.filterwarnings('ignore')
from tensorflow.keras.utils import to_categorical
from sklearn.preprocessing import LabelEncoder
import librosa
labelencoder=LabelEncoder()
extracted_features_df = pd.read_csv('data.csv')
X=np.array(extracted_features_df['feature'].tolist())
y=np.array(extracted_features_df['class'].tolist())
y=to_categorical(labelencoder.fit_transform(y))

app = Flask(__name__)
model = tf.keras.models.load_model('BGR_AUDIO')


def predict_gender(audio, sample_rate):
  mfccs_features = librosa.feature.mfcc(y=audio, sr=sample_rate, n_mfcc=40)
  mfccs_scaled_features = np.mean(mfccs_features.T,axis=0)
  mfccs_scaled_features=mfccs_scaled_features.reshape(1,-1)

  predicted_label=model.predict(mfccs_scaled_features, verbose=0)
  classes_x=np.argmax(predicted_label,axis=1)
  prediction_class = labelencoder.inverse_transform(classes_x)
  return prediction_class[0]


@app.route('/', methods=["GET", "POST"])
def index():
    gender = ""
    if request.method == "POST":
        print("FORM DATA RECEIVED")

        if "file" not in request.files:
            return redirect(request.url)

        file = request.files["file"]
        if file.filename == "":
            return redirect(request.url)

        if file:
            audio, sample_rate = librosa.load(file, res_type='kaiser_fast')
            gender = predict_gender(audio=audio, sample_rate=sample_rate)

    return render_template('index.html', gender=gender)

if __name__ == "__main__":
    app.run(host="0.0.0.0", port="5000")