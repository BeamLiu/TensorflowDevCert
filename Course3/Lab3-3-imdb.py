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
embedding_len = 32
tokenizer = Tokenizer(num_words=vocab_size, oov_token='<OOV>')
tokenizer.fit_on_texts(training_sentences)

training_sequenes = tokenizer.texts_to_sequences(training_sentences)
training_padded = pad_sequences(training_sequenes, maxlen=max_len, truncating='post')
print(f'traing data size: {len(training_sequenes)}')

testing_sequenes = tokenizer.texts_to_sequences(training_sentences)
testing_padded = pad_sequences(training_sequenes, maxlen=max_len)
print(f'testing data size: {len(testing_sequenes)}')

model = keras.Sequential([
    keras.layers.Embedding(vocab_size, embedding_len, input_length=max_len),
    keras.layers.Flatten(),
    keras.layers.Dense(6, activation=keras.activations.relu),
    keras.layers.Dense(1, activation=keras.activations.sigmoid)
])

model.compile(loss=keras.losses.binary_crossentropy, optimizer=keras.optimizers.Adam(), metrics=['accuracy'])

print(model.summary())

history = model.fit(training_padded, np.array(training_labels), epochs=10, verbose=1,
                    validation_data=(testing_padded, np.array(testing_labels)))

show_model_history(history)

# show the embeddings
embedding_layer = model.layers[0]
embedding_weights = embedding_layer.get_weights()[0]
print(embedding_weights)

# output tsv
with io.open('vecs.tsv', 'w', encoding='utf-8') as out_v:
    with io.open('meta.tsv', 'w', encoding='utf-8') as out_m:
        for word_num in range(1, vocab_size):
            word_name = tokenizer.index_word.get(word_num)
            word_embedding = embedding_weights[word_num]
            out_m.write(word_name + '\n')
            out_v.write('\t'.join([str(x) for x in word_embedding]) + '\n')

print('please manully go to https://projector.tensorflow.org/ to visualize the tsv files!')
