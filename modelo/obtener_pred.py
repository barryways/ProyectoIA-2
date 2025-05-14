import tensorflow as tf
import numpy as np
from tensorflow.keras.preprocessing import image
import pickle
import os

# Cargar modelo y etiquetas UNA SOLA VEZ al iniciar el servidor
model = tf.keras.models.load_model("asl_model.h5")

with open("labels.pkl", "rb") as f:
    class_indices = pickle.load(f)

# Diccionario invertido {índice: clase}
labels = {v: k for k, v in class_indices.items()}


def predecir_letra(ruta_imagen: str) -> str:
    """
    Procesa una imagen y devuelve la letra predicha por el modelo.

    :param ruta_imagen: Ruta a la imagen de entrada.
    :return: Letra predicha o mensaje de error.
    """
    if not os.path.exists(ruta_imagen):
        return "Error: Imagen no encontrada."

    try:
        img = image.load_img(ruta_imagen, target_size=(64, 64), color_mode='rgb')
        img_array = image.img_to_array(img)
        img_array = np.expand_dims(img_array, axis=0) / 255.0

        prediction = model.predict(img_array)
        predicted_class = np.argmax(prediction)

        letra_predicha = labels.get(predicted_class, "Clase no reconocida")
        return letra_predicha

    except Exception as e:
        return f"Error al procesar la imagen: {str(e)}"
