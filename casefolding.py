# Melakukan lower casing
df['kalimat'] = df['kalimat'].str.lower()
# Menghapus tanda baca
df['kalimat'] = df['kalimat'].str.replace('[^a-zA-Z0-9]', ' ')
# Menghilangkan karakter tunggal
df['kalimat'] = df['kalimat'].str.replace(r"\s+[a-zA-Z]\s+", ' ')
# Menghilagkan spasi ganda
df['kalimat'] = df['kalimat'].str.replace(r'[•\t|\n|\s+]', ' ')
