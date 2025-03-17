import nltk
from nltk.corpus import stopwords

# Download stopwords jika belum diunduh
nltk.download('stopwords')

# Load stopwords untuk bahasa Indonesia
stopwords_indonesian = stopwords.words('indonesian')

# Daftar kata yang tetap diperlukan meskipun ada dalam stopwords
needwords = ["tidak", "ada"]

# Menghapus kata-kata yang diperlukan dari stopwords
for word in needwords:
    if word in stopwords_indonesian:
        stopwords_indonesian.remove(word)

# Fungsi untuk menghapus stopwords dari teks
def remove_stopwords(text):
    # Pisahkan teks menjadi kata-kata dan hapus stopwords serta kata yang panjangnya 1
    text = [word for word in text.split() if len(word) > 1 and word.lower() not in stopwords_indonesian]
    # Gabungkan kata-kata menjadi teks kembali
    text = " ".join(text)
    return text
df['kalimat'] = df['kalimat'].apply(remove_stopwords)

