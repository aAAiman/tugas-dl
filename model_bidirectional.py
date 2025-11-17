import tensorflow as tf
from tensorflow.keras.preprocessing.sequence import pad_sequences
import numpy as np
import pickle

# Load tokenizer
with open("tokenizer.pkl", "rb") as f:
    tokenizer = pickle.load(f)

# Load panjang sequence
with open("max_seq_len.txt", "r") as f:
    max_seq_len = int(f.read())

# Load model Bidirectional LSTM
model = tf.keras.models.load_model("model_bidirectional.h5")

# Fungsi prediksi kata berikutnya
def run_bidirectional_model(text):
    token_list = tokenizer.texts_to_sequences([text])[0]

    token_list = pad_sequences([token_list], maxlen=max_seq_len - 1, padding='pre')

    predictions = model.predict(token_list, verbose=0)
    predicted_id = np.argmax(predictions)

    for word, index in tokenizer.word_index.items():
        if index == predicted_id:
            return word

    return "Tidak tahu kata berikutnya."
