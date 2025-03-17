from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences

# Inisiasi Tokenizer dengan jumlah kata maksimal (5000)
tokenizer = Tokenizer(num_words=5000)
tokenizer.fit_on_texts(X)

# Mengubah teks menjadi urutan angka (list of integers)
X_train = tokenizer.texts_to_sequences(X_train)
X_test = tokenizer.texts_to_sequences(X_test)

# Mendapatkan ukuran kosakata
vocab_size = len(tokenizer.word_index) + 1

# Menetapkan panjang maksimum untuk padding
maxlen = 100

# Mentranformasikan list angka menjadi array 2D dengan padding
X_train = pad_sequences(X_train, padding='post', maxlen=maxlen)
X_test = pad_sequences(X_test, padding='post', maxlen=maxlen)
