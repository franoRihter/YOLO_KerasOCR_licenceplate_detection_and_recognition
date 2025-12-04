import glob, os
from pathlib import Path
import tensorflow as tf
import matplotlib.pyplot as plt
from vocabolary import LabelConverter

import keras_ocr
label_converter = LabelConverter()

# Find all the images inside the folder (only the name)
validation_name = Path("ZG3833AK.jpg")

img_width = 200
img_height = 31
batch_size = 8


"""## Build and train the keras-ocr recognizer"""
build_params = keras_ocr.recognition.DEFAULT_BUILD_PARAMS 
build_params['width'] = img_width
build_params['height'] = img_height
# Version with custom vocabulary
recognizer = keras_ocr.recognition.Recognizer(alphabet=label_converter.lookup.get_vocabulary(), weights=None, build_params=build_params)
recognizer.prediction_model.load_weights("tocnost92posto.h5")
# print(recognizer.alphabet)


# print(validation_steps)

recognizer.compile()
labels = []
correct = 0
prediction = recognizer.recognize(validation_name.numpy())
#label = validation_name.numpy().decode("utf-8")
labels.append(validation_name)
print(prediction)
eq = prediction == label
if eq:
  correct += 1
else: 
  print(prediction, label, eq)
acc = correct / len(labels) * 100
