import numpy as np
import tensorflow as tf
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Embedding, Bidirectional, LSTM, Dense
import pickle

# Dataset teks sederhana (boleh ditambah)
corpus = [
    "saya suka makan nasi goreng",
    "saya suka belajar pemrograman",
    "dia sedang makan ayam bakar",
    "cuaca hari ini sangat panas",
    "aku ingin pergi ke kampus besok pagi",
    "kami sedang mengerjakan tugas kuliah",
    "mereka bermain bola di lapangan",
    "ibu sedang memasak di dapur",
    "ayah pergi bekerja setiap pagi",
    "adik sedang belajar membaca",
    "saya membeli kopi di minimarket",
    "temanku suka mendengarkan musik",
    "kami menonton film bersama",
    "dia sedang bermain game online",
    "saya ingin tidur lebih awal",
    "hari ini saya pergi ke supermarket",
    "harga bensin naik lagi",
    "banyak orang bekerja dari rumah",
    "laptop saya kehabisan baterai",
    "aku sedang menulis laporan",
    "kami belajar machine learning hari ini",
    "temanku belajar deep learning",
    "saya ingin menjadi data scientist",
    "kuliah online mulai jam delapan",
    "kami mengerjakan proyek kelompok",
    "saya sedang mengetik tugas",
    "guru menjelaskan materi dengan jelas",
    "kelas dimulai pada pukul delapan",
    "mobil itu melaju dengan cepat",
    "saya melihat berita di televisi",
    "pesawat itu lepas landas dengan mulus",
    "orang itu sedang membaca buku",
    "anak kecil itu sedang menangis",
    "kami menunggu bus di halte",
    "saya ingin makan bakso",
    "dia memesan mie ayam",
    "aku membeli jus jeruk",
    "kami makan malam bersama keluarga",
    "adik mandi sebelum tidur",
    "saya membeli sayur di pasar",
    "dia menonton pertandingan sepak bola",
    "temanku membeli smartphone baru",
    "saya bekerja sebagai programmer",
    "kami belajar algoritma dan struktur data",
    "dosennya menjelaskan teori jaringan",
    "saya sedang memperbaiki komputer",
    "kami menjalankan program python",
    "saya suka membaca artikel teknologi",
    "kami mendiskusikan proyek machine learning",
    "dia sedang mengetik laporan praktikum",
    "kami sedang melakukan penelitian",
    "saya suka kopi panas di pagi hari",
    "temanku membeli buku pemrograman",
    "kami berjalan-jalan di taman",
    "saya sedang membuat desain aplikasi",
]


# Tokenizer
tokenizer = Tokenizer()
tokenizer.fit_on_texts(corpus)
total_words = len(tokenizer.word_index) + 1

# Simpan tokenizer
with open("tokenizer.pkl", "wb") as f:
    pickle.dump(tokenizer, f)

# Membuat input sequences
input_sequences = []
for line in corpus:
    token_list = tokenizer.texts_to_sequences([line])[0]
    for i in range(1, len(token_list)):
        ngram = token_list[:i+1]
        input_sequences.append(ngram)

# Padding
max_seq_len = max([len(x) for x in input_sequences])
input_sequences = np.array(pad_sequences(input_sequences, maxlen=max_seq_len, padding='pre'))

# X dan y
X = input_sequences[:, :-1]
y = tf.keras.utils.to_categorical(input_sequences[:, -1], num_classes=total_words)

# ===== MODEL BIDIRECTIONAL LSTM =====
model = Sequential()
model.add(Embedding(total_words, 128, input_length=max_seq_len - 1))
model.add(Bidirectional(LSTM(128)))
model.add(Dense(total_words, activation='softmax'))

model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
model.summary()

model.fit(X, y, epochs=200, verbose=1)

# Simpan model
model.save("model_bidirectional.h5")

# Simpan panjang sequence
with open("max_seq_len.txt", "w") as f:
    f.write(str(max_seq_len))
