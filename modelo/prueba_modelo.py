import tensorflow as tf
import numpy as np
from tensorflow.keras.preprocessing import image
import pickle

# Cargar el modelo
model = tf.keras.models.load_model("asl_model.h5")

# Cargar el diccionario de clases
with open("labels.pkl", "rb") as f:
    class_indices = pickle.load(f)

# Invertir el diccionario para obtener {índice: clase}
labels = {v: k for k, v in class_indices.items()}

# Ruta a la imagen de prueba
image_path = "asl/asl_alphabet_test/asl_alphabet_test/D_test.jpg"  # Cambia esto según tu imagen

# Preprocesar la imagen
img = image.load_img(image_path, target_size=(64, 64), color_mode='rgb')
img_array = image.img_to_array(img)
img_array = np.expand_dims(img_array, axis=0)
img_array = img_array / 255.0

# Predicción
prediction = model.predict(img_array)
predicted_class = np.argmax(prediction)

# Mostrar resultado
letra_predicha = labels[predicted_class]
print(f"La letra es: {letra_predicha}")
