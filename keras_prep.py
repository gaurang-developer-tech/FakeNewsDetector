import pandas as pd
import pickle
import numpy as np
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
df=pd.read_csv('Cleaned_News_dl.csv')
df.dropna(inplace=True)

X_text=df['content'].astype(str)
y=df['label'].values

max_words=10000
max_sequence_len=500

tokenizer=Tokenizer(num_words=max_words)
tokenizer.fit_on_texts(X_text)

sequences=tokenizer.texts_to_sequences(X_text)
X_padded=pad_sequences(sequences, maxlen=max_sequence_len,padding='post',truncating='post')

with open('tokenizer.pkl', 'wb') as handle:
    pickle.dump(tokenizer, handle, protocol=pickle.HIGHEST_PROTOCOL)

np.save('X_padded.npy', X_padded)
np.save('y_labels.npy', y)

print(f"Success!Input matrix shape: {X_padded.shape}")
