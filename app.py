import os
from flask import Flask, render_template, request
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
import numpy as np

app = Flask(__name__)

# Load model yang telah disimpan
model = load_model('NASNetMobile.h5')

# Daftar kelas
class_labels = {
    0: "Ada hama",
    1: "Tanaman sehat"
}

def prepare_image(img_path, target_size):
    img = image.load_img(img_path, target_size=target_size)
    img_array = image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)
    img_array /= 255.0
    return img_array

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    if request.method == 'POST':
        if 'file' not in request.files:
            return "File tidak ditemukan", 400

        file = request.files['file']
        if file.filename == '':
            return "Nama file kosong", 400

        file_path = os.path.join('uploads', file.filename)
        file.save(file_path)

        img_array = prepare_image(file_path, target_size=(128, 128))

        predictions = model.predict(img_array)
        predicted_class = np.argmax(predictions)
        confidence = np.max(predictions)
        predicted_class_label = class_labels[predicted_class]

        return render_template('result.html', predicted_class=predicted_class_label, confidence=confidence, file=file_path)

if __name__ == '__main__':
    app.run(debug=True)
