from tensorflow.keras.preprocessing import image
from tensorflow.keras.models import load_model
from tensorflow.keras.applications.inception_resnet_v2 import preprocess_input
import numpy as np

model = load_model(r'Models/melhor_modelo.keras')
classes = ['A', 'B', 'C','D','E','I','L','M','N','O','R','S','U','V','W']  # sua lista de classes
IMG_SIZE = (224, 224)

img_path = r"C:\Users\Allisson\Desktop\Semtitulo.png"
img = image.load_img(img_path, target_size=IMG_SIZE)
img_array = image.img_to_array(img)
img_array = preprocess_input(img_array)
img_array = np.expand_dims(img_array, axis=0)

pred = model.predict(img_array)
label = classes[np.argmax(pred)]

print(f'Letra prevista: {label}')
