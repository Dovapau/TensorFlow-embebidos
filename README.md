# 🧠 Tutorial Guíado: TensorFlow Lite con Python

## 🎯 Objetivo
Instalar TensorFlow Lite y ejecutar un modelo de ejemplo de clasificación de imágenes desde Python. Este tutorial está pensado para principiantes y dispositivos con recursos limitados (como Raspberry Pi).

---

## 🧰 Requisitos

- ✅ Python 3.8 a 3.12 instalado
- ✅ Acceso a terminal o consola
- ✅ Conexión a internet
- ✅ Funciona en Windows, Linux, Raspberry Pi OS

---

## ✅ Paso 1: Crear un entorno virtual (opcional pero recomendado)

### En Windows / Linux / Raspberry Pi:

```bash
python -m venv tflite_env
```

Activamos el entorno:

**En Windows**

```bash
tflite_env\Scripts\activate
```

**En Linux**

```bash
source tflite_env/bin/activate
```

## ✅ Paso 2: Instalar TensorFlow Lite

Instalamos el paquete correspondiente:

```bash
pip install tflite-runtime
```

## ✅ Paso 3: Descargamos el modelo preentrenado

Abrimos NANO y creamos un script de Python:

```bash
nano download_model.py
```

Dentro de NANO escribimos:

```python
import urllib.request

MODEL_URL = "https://storage.googleapis.com/download.tensorflow.org/models/tflite/model_mobilenet_v1_1.0_224_quant.tflite"
MODEL_PATH = "mobilenet_v1.tflite"

urllib.request.urlretrieve(MODEL_URL, MODEL_PATH)

print("Modelo descargado como mobilenet_v1.tflite")
```
Ejecutamos:

```bash
python download_model.py
```

## ✅ Paso 4: Descargamos el modelo preentrenado

Abrimos NANO y creamos un script de Python:

```bash
nano download_image.py
```

Dentro de NANO escribimos:

```python
import urllib.request

IMAGE_URL = "https://tensorflow.org/images/grace_hopper.jpg"
urllib.request.urlretrieve(IMAGE_URL, "grace_hopper.jpg")

print("Imagen descargada")
```

Ejecutamos:

```bash
python download_image.py
```
## ✅ Paso 5: Ejecutar el modelo con TensorFlow Lite

Abrimos NANO y creamos un script de Python:

```bash
nano run_model.py
```
Dentro de NANO escribimos:

```python
import numpy as np
from PIL import Image
import tflite_runtime.interpreter as tflite

# Cargar modelo
interpreter = tflite.Interpreter(model_path="mobilenet_v1.tflite")
interpreter.allocate_tensors()

input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()

# Preparar imagen
img = Image.open("grace_hopper.jpg").resize((224, 224))
input_data = np.expand_dims(np.array(img, dtype=np.uint8), axis=0)

# Inferencia
interpreter.set_tensor(input_details[0]['index'], input_data)
interpreter.invoke()
output = interpreter.get_tensor(output_details[0]['index'])

# Mostrar resultado
top_index = np.argmax(output)
print(f"Predicción (índice): {top_index}")
```

Ejecutamos:

```bash
python run_model.py
```

Resultado Final:

Después de ejecutar todos los pasos debemos obtener un número en la consola por ejemplo:

```scss
Predicción (índice): 653
```

Este número representa la categoría predicha por el modelo y comprueba el funcionamiento adecuado de la herramienta.


# Opcional: Etiequetas reales: 

Abre en terminal el siguiente paquete:

```bash
wget https://storage.googleapis.com/download.tensorflow.org/data/imagenet_labels.txt
```
Luego abre el archivo *imagenet_labels.txt* y busca la línea con el índice.


## 📝 Consejos

- ✅ Asegúrate de tener activado el entorno virtual al instalar o ejecutar los scripts.
- 📂 Todos los archivos `.py`, `.tflite` e imagen deben estar en la **misma carpeta**.
- 🛠️ Si algo falla, repite el paso anterior o revisa si hubo errores en la consola.

---

## 📚 ¿Qué aprendiste?

- 📦 Instalar TensorFlow Lite en un entorno liviano  
- 🔍 Descargar y usar un modelo preentrenado  
- 🖼️ Ejecutar inferencia en una imagen  
- 🚫 Todo sin necesidad de GPU o conexión a la nube









