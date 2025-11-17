import tensorflow as tf
import numpy as np

# Load model sekali saja
model = tf.keras.applications.MobileNetV2(weights="imagenet")

# Preprocessing function
def prepare_image(img_path):
    img = tf.keras.preprocessing.image.load_img(img_path, target_size=(224, 224))
    img_array = tf.keras.preprocessing.image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)
    img_array = tf.keras.applications.mobilenet_v2.preprocess_input(img_array)
    return img_array

# Prediction function
def classify_image(img_path):
    img = prepare_image(img_path)
    predictions = model.predict(img)

    decoded = tf.keras.applications.mobilenet_v2.decode_predictions(predictions, top=1)

    # Contoh hasil: [[('n02124075', 'Egyptian_cat', 0.97)]]
    label = decoded[0][0][1].replace("_", " ")

    return label
