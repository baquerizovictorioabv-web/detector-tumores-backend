import os
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
from flask import Flask, jsonify, request
from flask_cors import CORS
from tensorflow.keras.preprocessing import image
from werkzeug.utils import secure_filename

matplotlib.use('Agg')

app = Flask(__name__)
CORS(app)

# Configuración del modelo TFLite
interpreter = tf.lite.Interpreter(model_path='brain_tumor_cnn.tflite')
interpreter.allocate_tensors()

input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()

class_names = ['glioma', 'meningioma', 'notumor', 'pituitary']

# Configuración de la carpeta de subidas
UPLOAD_FOLDER = os.path.join('static', 'uploads')
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER


def predict_with_tflite(img_array):
    img_array = img_array.astype(np.float32) / 255.0

    interpreter.set_tensor(input_details[0]['index'], img_array)
    interpreter.invoke()

    output_data = interpreter.get_tensor(output_details[0]['index'])
    return output_data[0]


@app.route('/api/clasificar', methods=['POST'])
def clasificar_api():
    file = request.files.get('image')

    if not file:
        return jsonify({'error': 'No se envió imagen'}), 400

    filename = secure_filename(file.filename)
    filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
    file.save(filepath)

    # Cargar y preprocesar la imagen
    img = image.load_img(filepath, target_size=(128, 128))
    img_array = image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)

    # Predicción
    prediction = predict_with_tflite(img_array)
    predicted_class = class_names[np.argmax(prediction)]

    probabilities = {
        class_names[i]: float(f"{prob:.4f}")
        for i, prob in enumerate(prediction)
    }

    # Generar gráfico de barras
    plt.figure(figsize=(6, 4))
    plt.bar(probabilities.keys(), probabilities.values(), color='skyblue')
    plt.title('Probabilidades por clase')
    plt.ylabel('Confianza')
    plt.tight_layout()

    graph_path = os.path.join(app.config['UPLOAD_FOLDER'], 'probabilidades.png')
    plt.savefig(graph_path)
    plt.close()

    return jsonify({
        'prediction': f'Predicción: {predicted_class.upper()}',
        'image_name': filename,
        'probs': probabilities
    })


if __name__ == '__main__':
    app.run(debug=True, port=5000)