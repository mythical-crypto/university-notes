# Демонстрация классификатора в браузере (после обучения модели в Colab)
from pathlib import Path
import numpy as np
from PIL import Image
import gradio as gr
import tensorflow as tf

BASE = Path(__file__).resolve().parent
file = BASE/'artifacts'/'mobilenetv2_cifar3.keras'
if not file.exists():
    raise FileNotFoundError('Нет обученной модели: сначала запустите ноутбук и скопируйте .keras в папку artifacts')
model = tf.keras.models.load_model(str(file), compile=False)
NAMES=['bird','cat','dog']

def predict(img):
    arr=np.asarray(img.convert('RGB').resize((96,96))).astype(np.float32)
    prob=model.predict(arr[None,...],verbose=0)[0]
    return {name:float(p) for name,p in zip(NAMES,prob)}

gr.Interface(fn=predict, inputs=gr.Image(type='pil',label='Изображение'),
             outputs=gr.Label(num_top_classes=3),
             title='CIFAR-10: bird / cat / dog',
             description='MobileNetV2 transfer learning. Демонстрация после обучения модели.').launch()