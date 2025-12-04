import os
import tensorflow as tf
from pathlib import Path
import keras_ocr
from vocabolary import LabelConverter

# --- Setup ---
label_converter = LabelConverter()

img_width = 200
img_height = 31

# Create recognizer
build_params = keras_ocr.recognition.DEFAULT_BUILD_PARAMS
build_params['width'] = img_width
build_params['height'] = img_height

recognizer = keras_ocr.recognition.Recognizer(
    alphabet=label_converter.lookup.get_vocabulary(),
    weights=None,
    build_params=build_params
)
recognizer.prediction_model.load_weights("recognizer_borndigital.h5")

# --- Choose ONE IMAGE ---
image_path = "ZG3833AK.jpg"   # <--- put one filename here


# --- Preprocess function
def load_and_preprocess(path):
    img = tf.io.read_file(path)
    img = tf.image.decode_jpeg(img, channels=3)
    img = tf.image.resize(
        img,
        [img_height, img_width],
        method=tf.image.ResizeMethod.NEAREST_NEIGHBOR
    )
    return img

# Load image
img = load_and_preprocess(image_path)

# Convert to NumPy
img_np = img.numpy()

# --- Get label from filename (same logic you use) ---
filename = Path(image_path).name
label = filename.split(".jpg")[0].split("_")[0].split(" ")[0].upper()

# --- Predict ---
prediction = recognizer.recognize(img_np)

print("Image Path:", image_path)
print("True Label:", label)
print("Prediction:", prediction)