# Healthy Teeth Artificial Intelligence

## Descripción
Healthy Teeth AI es una herramienta de diagnóstico automático de enfermedades bucodentales. Su principal objetivo es simplificar el proceso de diagnóstico odontológico, detectando de manera precisa y rápida afecciones bucales comunes como caries, gingivitis y úlceras.

## Casos de Uso y Aplicaciones Clínicas
- **Detección Temprana:** El sistema puede detectar problemas dentales en etapas iniciales, permitiendo un tratamiento más efectivo.
- **Diagnóstico Asistido:** Los odontólogos pueden utilizar el modelo para respaldar sus diagnósticos y mejorar la precisión.
- **Eficiencia Clínica:** La automatización del proceso de reconocimiento de problemas dentales mejora la productividad y reduce los tiempos de espera.
- **Educación Médica:** El sistema puede utilizarse en la formación de futuros odontólogos, proporcionando una herramienta de aprendizaje valiosa.

## Conjunto de Datos de Entrenamiento
- **Amplia Variedad:** El conjunto de datos incluye 1500 fotografías de una amplia gama de problemas dentales, desde caries hasta enfermedades de las encías.
- **Alta Calidad:** Las imágenes han sido tomadas con equipos profesionales y cuentan con una excelente resolución y nitidez.
- **Anotaciones Expertas:** Cada fotografía ha sido revisada y anotada por odontólogos expertos, garantizando la precisión del diagnóstico.
- **Fuente de las Imágenes:** Las imágenes fueron obtenidas del conjunto de datos [Oral Diseases](https://www.kaggle.com/datasets/salmansajid05/oral-diseases?resource=download) disponible en Kaggle.

## Entrenamiento y Validación del Modelo
1. **División de Datos:** El conjunto de 1500 imágenes se divide en subconjuntos de entrenamiento, validación y prueba.
2. **Entrenamiento:** El modelo se entrena iterativamente utilizando el subconjunto de entrenamiento, optimizando los parámetros para mejorar la precisión.
3. **Validación:** El rendimiento del modelo se evalúa continuamente utilizando el subconjunto de validación para evitar el sobreajuste.

## Tecnologías Usadas
- **Lenguajes y Bibliotecas:** Python, TensorFlow, Keras, Flask, NumPy, Pandas
- **Herramientas de Desarrollo:** Jupyter Notebook, PyCharm, Visual Studio Code
- **Servicios de Almacenamiento:** Google Drive (para almacenar y compartir el conjunto de datos)
  
## Conclusiones y Próximos Pasos
- **Avances Notables:** El sistema de IA para reconocimiento de problemas dentales ha demostrado ser altamente preciso y eficiente.
- **Próximas Mejoras:** Se planea expandir el conjunto de datos y refinar aún más el modelo para mejorar su desempeño.
- **Implementación Clínica:** El objetivo es integrar esta tecnología en entornos odontológicos para beneficiar a los pacientes y profesionales.

## Integrantes del Grupo
- Branly Peña Lizardo
- Awilka Jerome Puente
- Erick Octavio Rivera
- Dawny Chalas Telemaco

## Muestras del Proyecto

![Interfaz del proyecto](https://github.com/DawnyCTI/PROYECTO-FINAL---GRUPO-STAT-CAST/raw/main/images/p1.png)
*Figura 1: Interfaz del proyecto*

![Selección de imagen con problema](https://github.com/DawnyCTI/PROYECTO-FINAL---GRUPO-STAT-CAST/raw/main/images/p2.png)
*Figura 2: Selección de imagen con problema*

![Resultado del diagnóstico](https://github.com/DawnyCTI/PROYECTO-FINAL---GRUPO-STAT-CAST/raw/main/images/p3.png)
*Figura 3: Resultado del diagnóstico*

## Ejemplo de Uso

## Entrenamiento del Modelo
```python
import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator

# Directorio donde se encuentran las imágenes categorizadas
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

# Obtener las etiquetas de las clases
class_labels = list(train_generator.class_indices.keys())
print("Etiquetas de las clases:", class_labels)

# Definir la arquitectura del modelo
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

# Compilación del modelo
model.compile(
    loss='categorical_crossentropy',
    optimizer='adam',
    metrics=['accuracy']
)

# Entrenamiento del modelo
model.fit(train_generator, epochs=10)

# Evaluación del modelo
loss, accuracy = model.evaluate(train_generator)
print(f"Pérdida: {loss}, Precisión: {accuracy}")

# Guardado del modelo
model.save('modelo_entrenado.h5')

# -------------------------------
# Fin del entrenamiento del modelo
