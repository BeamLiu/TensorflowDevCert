import io

import keras
import numpy as np
from keras.src.preprocessing.text import Tokenizer
from keras.src.utils import pad_sequences

from Course3.Utility import show_model_history

# Read the data
with open('./dataset/sonnets.txt') as f:
    data = f.read()

corpus = data.lower().split('\n')
print(corpus)

tokenizer = Tokenizer()
tokenizer.fit_on_texts(corpus)

total_words = len(tokenizer.word_index) + 1
print(f'total words: {total_words}')
print(f'index: {tokenizer.word_index}')

input_sequences = []
for line in corpus:
    token_list = tokenizer.texts_to_sequences([line])[0]
    for idx in range(1, len(token_list)):
        # create the subphrase
        gram_sequence = token_list[:idx + 1]
        input_sequences.append(gram_sequence)

max_len = max([len(x) for x in input_sequences])
input_sequences = np.array(pad_sequences(input_sequences, maxlen=max_len, padding='pre'))
xs, labels = input_sequences[:, :-1], input_sequences[:, -1]
print(xs)
ys = keras.utils.to_categorical(labels, num_classes=total_words)

sentences = corpus[0].split()
print((f'sample sentences: {sentences}'))

model = keras.Sequential([
    keras.layers.Embedding(total_words, 64, input_length=max_len - 1),
    keras.layers.Bidirectional(keras.layers.LSTM(20)),
    keras.layers.Dense(total_words, activation=keras.activations.softmax)
])

print(model.summary())

model.compile(loss=keras.losses.categorical_crossentropy, optimizer=keras.optimizers.Adam(), metrics=['accuracy'])
history = model.fit(xs, np.array(ys), epochs=10, verbose=1)
show_model_history(history)

# Define seed text
seed_text = "help me obi-wan kinobi youre my only hope"
# Define total words to predict
next_words = 100
# Loop until desired length is reached
for _ in range(next_words):
    # Convert the seed text to a token sequence
    token_list = tokenizer.texts_to_sequences([seed_text])[0]
    # Pad the sequence
    token_list = pad_sequences([token_list], maxlen=max_len - 1, padding='pre')
    # Feed to the model and get the probabilities for each index
    probabilities = model.predict(token_list)
    # Pick a random number from [1,2,3]
    choice = np.random.choice([1, 2, 3])
    # Sort the probabilities in ascending order
    # and get the random choice from the end of the array
    predicted = np.argsort(probabilities)[0][-choice]
    # Ignore if index is 0 because that is just the padding.
    if predicted != 0:
        # Look up the word associated with the index.
        output_word = tokenizer.index_word[predicted]
        # Combine with the seed text
        seed_text += " " + output_word
# Print the result
print(seed_text)
