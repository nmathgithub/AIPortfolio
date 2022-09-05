from msilib import datasizemask
import tensorflow as tf 
from tensorflow import keras
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences

# Tutorial (YouTube)
# Natural Language Process - Tokenization (NLP Zero to Hero - Part I)
# 5 September 2022 

# Training Data (basic example)
sentences = [
    'I love my dog',
    'I love my cat',
    'You love my dog!', 
    'Do you think my dog is amazing?'
]



tokenizer = Tokenizer(num_words = 100, oov_token = "<OOV>")
tokenizer.fit_on_texts(sentences)
word_index = tokenizer.word_index 

sequences = tokenizer.texts_to_sequences(sentences)

padded = pad_sequences(sequences, padding = 'post', 
                       truncating = 'post', maxlen=5)
print(word_index)
print(sequences)
print(padded)
# Test Data 
test_data = [
    'I really love my dog', 
    'my dog loves my manatee'
]

test_seq = tokenizer.texts_to_sequences(test_data)
print(test_seq)