from flask import Flask, render_template, request
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
import numpy as np
import os

app = Flask(__name__)
model = load_model("C:/Users/dawny/Desktop/PROYECTO FINAL - GRUPO STAT CAST/Interfaz - proyecto-IA/modelo_entrenado.h5")
class_labels = ['Calculus', 'Gingivitis', 'Mouth Ulcer', 'Tooth Discoloration', 'Hypodontia', 'Healthy']

# Recomendaciones específicas para cada clase
recommendations = {
    'Calculus': 'Es importante visitar a tu dentista para una limpieza profesional. Mantén una buena higiene dental y usa hilo dental regularmente.',
    'Gingivitis': 'Cepíllate y usa hilo dental con más frecuencia. Considera usar un enjuague bucal antibacteriano y visita a tu dentista para un tratamiento profesional.',
    'Mouth Ulcer': 'Evita alimentos ácidos o picantes. Usa un enjuague bucal suave y, si la úlcera persiste, consulta a tu dentista o médico.',
    'Tooth Discoloration': 'Limita el consumo de café, té y vino tinto. Considera un tratamiento de blanqueamiento dental y mantén una buena higiene bucal.',
    'Hypodontia': 'Consulta con tu dentista sobre posibles opciones de tratamiento, como implantes dentales o puentes, para reemplazar los dientes faltantes.',
    'Healthy': '¡Tus dientes parecen saludables! Sigue con tu rutina de cuidado dental diaria y visita a tu dentista regularmente para chequeos.'
}

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    if request.method == 'POST':
        img = request.files['image']
        if img:
            img_dir = 'C:/Users/dawny/Downloads/interfaz/interfaz/images'
            if not os.path.exists(img_dir):
                os.makedirs(img_dir)

            img_path = os.path.join(img_dir, img.filename)
            img.save(img_path)

            img = image.load_img(img_path, target_size=(150, 150))
            img_array = image.img_to_array(img)
            img_array = np.expand_dims(img_array, axis=0)
            img_array /= 255.

            prediction = model.predict(img_array)
            predicted_class = class_labels[np.argmax(prediction)]
            confidence = round(np.max(prediction) * 100, 2)

            recommendation = recommendations.get(predicted_class, 'Consulta a tu dentista para más información.')

            return render_template('result.html', prediction=predicted_class, confidence=confidence, recommendation=recommendation)
        else:
            return "No image uploaded", 400

if __name__ == '__main__':
    app.run(debug=True)
