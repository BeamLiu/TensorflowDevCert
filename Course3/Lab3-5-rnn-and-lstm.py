import io

import keras
import numpy as np
import tensorflow_datasets as tfd
from keras.src.preprocessing.text import Tokenizer
from keras.src.utils import pad_sequences

from Course3.Utility import show_model_history

imdb, info = tfd.load('imdb_reviews', with_info=True, as_supervised=True)
print(info)

for example in imdb['train'].take(2):
    print(example)

train_data, test_data = imdb['train'], imdb['test']

training_sentences = []
training_labels = []

testing_sentences = []
testing_labels = []

for s, l in train_data:
    training_sentences.append(s.numpy().decode('utf-8'))
    training_labels.append(l.numpy())

for s, l in test_data:
    testing_sentences.append(s.numpy().decode('utf-8'))
    testing_labels.append(l.numpy())

vocab_size = 10000
max_len = 120
embedding_len = 16
epochs = 20
tokenizer = Tokenizer(num_words=vocab_size, oov_token='<OOV>')
tokenizer.fit_on_texts(training_sentences)

training_sequenes = tokenizer.texts_to_sequences(training_sentences)
training_padded = pad_sequences(training_sequenes, maxlen=max_len, truncating='post')
print(f'traing data size: {len(training_sequenes)}')

testing_sequenes = tokenizer.texts_to_sequences(testing_sentences)
testing_padded = pad_sequences(testing_sequenes, maxlen=max_len)
print(f'testing data size: {len(testing_sequenes)}')


def try_model(model, name):
    model.compile(loss=keras.losses.binary_crossentropy, optimizer=keras.optimizers.Adam(), metrics=['accuracy'])
    print(model.summary())
    history = model.fit(training_padded, np.array(training_labels), epochs=epochs, verbose=1,
                        validation_data=(testing_padded, np.array(testing_labels)))
    show_model_history(history, name)


embedding_model = keras.Sequential([
    keras.layers.Embedding(vocab_size, embedding_len, input_length=max_len),
    keras.layers.Flatten(),
    keras.layers.Dense(6, activation=keras.activations.relu),
    keras.layers.Dense(1, activation=keras.activations.sigmoid)
])
try_model(embedding_model, 'Embedding')

gru_model = keras.Sequential([
    keras.layers.Embedding(vocab_size, embedding_len, input_length=max_len),
    keras.layers.Bidirectional(keras.layers.GRU(32)),
    keras.layers.Dense(6, activation=keras.activations.relu),
    keras.layers.Dense(1, activation=keras.activations.sigmoid)
])
try_model(gru_model, 'GRU')

lstm_model = keras.Sequential([
    keras.layers.Embedding(vocab_size, embedding_len, input_length=max_len),
    keras.layers.Bidirectional(keras.layers.LSTM(64)),
    keras.layers.Dense(6, activation=keras.activations.relu),
    keras.layers.Dense(1, activation=keras.activations.sigmoid)
])
try_model(lstm_model, 'LSTM')