import tensorflow as tf 
from tensorflow import keras
from tensorflow.keras.preprocessing.text import Tokenizer

# Tutorial (YouTube)
# Natural Language Process - Tokenization (NLP Zero to Hero - Part I)
# 5 September 2022 

sentences = [
    'I love my dog',
    'I love my cat',
    'You love my dog!'
]

tokenizer = Tokenizer(num_words = 100)
tokenizer.fit_on_texts(sentences)
word_index = tokenizer.word_index 
print(word_index)