import pandas as pd
from Sastrawi.Stemmer.StemmerFactory import StemmerFactory

# Load colloquial-indonesian-lexicon.csv
#lexicon_df = pd.read_csv('path_to_your_lexicon.csv')
lexicon_df = pd.read_csv('/content/colloquial-indonesian-lexicon.csv')

# Create a dictionary for replacing colloquial words with formal ones
colloquial_words = dict(zip(lexicon_df['slang'], lexicon_df['formal']))

# Function to replace colloquial words with formal words
def replace_colloquial(text):
    words = text.split()
    replaced_words = [colloquial_words[word] if word in colloquial_words else word for word in words]
    return ' '.join(replaced_words)

# Initialize the Sastrawi Stemmer
factory = StemmerFactory()
stemmer = factory.create_stemmer()

# Function to apply stemming
def stem_text(text):
    return stemmer.stem(text)

# Apply the function to the 'pertanyaan' column to replace colloquial words
df['kalimat'] = df['kalimat'].apply(replace_colloquial)

# Apply the stemming function to the 'pertanyaan' column
df['kalimat'] = df['kalimat'].apply(stem_text)

# Display the first few rows of the updated DataFrame
print(df.head())

