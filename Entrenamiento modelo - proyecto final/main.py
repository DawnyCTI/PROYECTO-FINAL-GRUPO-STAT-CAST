
# Directorio de entrenamiento
import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator

#aqui poner el directorio de donde estan las carpetas con las imagenes en categorias
train_dir = 'C:/Users/branl/Desktop/samsung/proyectofinal'
# Configuración del generador de datos de imagen
train_datagen = ImageDataGenerator(
    rescale=1./255,
    shear_range=0.2,
    zoom_range=0.2,
    horizontal_flip=True
)

train_generator = train_datagen.flow_from_directory(
    train_dir,
    target_size=(150, 150),
    batch_size=32,
    class_mode='categorical'
)

# aqui obtiene las los nombres de las enfermedades y las organiza como estan en la carpeta
class_labels = list(train_generator.class_indices.keys())

print("Etiquetas de las clases:", class_labels)

# aqui se define la arquitectura del modelo
model = tf.keras.models.Sequential([
    tf.keras.layers.Conv2D(32, (3, 3), activation='relu', input_shape=(150, 150, 3)),
    tf.keras.layers.MaxPooling2D((2, 2)),
    tf.keras.layers.Conv2D(64, (3, 3), activation='relu'),
    tf.keras.layers.MaxPooling2D((2, 2)),
    tf.keras.layers.Conv2D(128, (3, 3), activation='relu'),
    tf.keras.layers.MaxPooling2D((2, 2)),
    tf.keras.layers.Flatten(),
    tf.keras.layers.Dense(512, activation='relu'),
    tf.keras.layers.Dense(len(class_labels), activation='softmax')
])
#aqui compilamos el modelo
model.compile(
    loss='categorical_crossentropy',
    optimizer='adam',
    metrics=['accuracy']
)

# aqui se entrena el modelo
model.fit(train_generator, epochs=10)

# evaluamos el modelo diciendo si el esta correctamente y si esta dando buenas predicciones
loss, accuracy = model.evaluate(train_generator)
print(f"Pérdida: {loss}, Precisión: {accuracy}")

#guardamos nuestro modelo
model.save('modelo_entrenado.h5')
