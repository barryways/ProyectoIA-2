import pickle
from tensorflow.keras.preprocessing.image import ImageDataGenerator

datagen = ImageDataGenerator(validation_split=0.2)

train_gen = datagen.flow_from_directory(
    'asl/asl_alphabet_train/asl_alphabet_train',
    target_size=(64, 64),
    batch_size=64,
    class_mode='categorical',
    subset='training',
    shuffle=False
)

# Guardar el mapeo de clases
with open("labels.pkl", "wb") as f:
    pickle.dump(train_gen.class_indices, f)

print("Diccionario de clases guardado como labels.pkl")
