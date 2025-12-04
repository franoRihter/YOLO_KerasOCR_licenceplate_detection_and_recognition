import tkinter as tk
from tkinter import filedialog
from PIL import Image, ImageTk
import tensorflow as tf
from pathlib import Path
from vocabolary import LabelConverter
import keras_ocr


# ------------ Load OCR model once ------------
label_converter = LabelConverter()
img_width = 200
img_height = 31

build_params = keras_ocr.recognition.DEFAULT_BUILD_PARAMS
build_params["width"] = img_width
build_params["height"] = img_height

recognizer = keras_ocr.recognition.Recognizer(
    alphabet=label_converter.lookup.get_vocabulary(),
    weights=None,
    build_params=build_params
)
recognizer.prediction_model.load_weights("recognizer_borndigital.h5")


# ------------ OCR INFERENCE ------------
def inference(path):
    def load_and_preprocess(path):
        img = tf.io.read_file(path)
        img = tf.image.decode_jpeg(img, channels=3)
        img = tf.image.resize(img, [img_height, img_width],
                              method=tf.image.ResizeMethod.NEAREST_NEIGHBOR)
        return img

    img = load_and_preprocess(path)
    img_np = img.numpy()
    prediction = recognizer.recognize(img_np)
    return prediction


# ------------ GUI FUNCTIONS ------------
def browse_file():
    file_path = filedialog.askopenfilename(
        title="Select a file",
        filetypes=[("Images", "*.jpg *.jpeg *.png"), ("All files", "*.*")]
    )

    if file_path:
        selected_file.set(file_path)
        entry.delete(0, tk.END)
        entry.insert(0, file_path)
        label_selected.config(text=f"Selected: {file_path}")
        show_image(file_path)


def show_image(path):
    img = Image.open(path)

    # Resize preview for bottom display
    img = img.resize((300, 200), Image.LANCZOS)

    tk_img = ImageTk.PhotoImage(img)

    # Must keep reference to avoid garbage collection
    image_label.image = tk_img
    image_label.configure(image=tk_img)


def run_prediction():
    path = selected_file.get()
    if not path:
        label_prediction.config(text="No file selected!")
        return

    pred = inference(path)
    label_prediction.config(text=f"Prediction: {pred}")


# ------------ GUI WINDOW ------------
root = tk.Tk()
root.title("OCR Tool")
root.geometry("600x500")     # taller window for bottom image
root.attributes("-topmost", True)

selected_file = tk.StringVar()

# TOP FRAME (text + buttons)
top_frame = tk.Frame(root)
top_frame.pack(side="top", padx=10, pady=10)

# BOTTOM FRAME (image)
bottom_frame = tk.Frame(root)
bottom_frame.pack(side="bottom", pady=10)

# Widgets in TOP frame
entry = tk.Entry(top_frame, width=40)
entry.pack(pady=(10, 5))

browse_btn = tk.Button(top_frame, text="Browse", command=browse_file)
browse_btn.pack(pady=4)

predict_btn = tk.Button(top_frame, text="Predict", command=run_prediction)
predict_btn.pack(pady=4)

label_selected = tk.Label(top_frame, text="Select file")
label_selected.pack(pady=6)

label_prediction = tk.Label(top_frame, text="Prediction:")
label_prediction.pack(pady=8)

# Image appears here at the bottom
image_label = tk.Label(bottom_frame)
image_label.pack()

root.mainloop()