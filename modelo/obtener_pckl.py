from tensorflow.keras.preprocessing.image import ImageDataGenerator

# Usamos los mismos parámetros que en el entrenamiento
DATASET_PATH = 'asl/asl_alphabet_train/asl_alphabet_train'
img_height, img_width = 64, 64
batch_size = 64

datagen = ImageDataGenerator(validation_split=0.2)

train_gen = datagen.flow_from_directory(
    DATASET_PATH,
    target_size=(img_height, img_width),
    batch_size=batch_size,
    class_mode='categorical',
    subset='training',
    shuffle=False  # <-- importante para ver el orden fijo
)

# Mostrar el mapeo de clase -> índice
print(train_gen.class_indices)

# Opcional: guardar para usar luego sin reentrenar
import pickle
with open("class_indices.pkl", "wb") as f:
    pickle.dump(train_gen.class_indices, f)
