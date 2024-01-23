import keras
import numpy as np
import tensorflow_datasets as tfd
from keras.src.utils import pad_sequences

from Course3.Utility import show_model_history

imdb, info = tfd.load('imdb_reviews/subwords8k', with_info=True, as_supervised=True)
# Get the encoder
tokenizer_subword = info.features['text'].encoder
print(tokenizer_subword.subwords)

for example in imdb['train'].take(2):
    print(example)

train_data, test_data = imdb['train'], imdb['test']

BUFFER_SIZE = 10000
BATCH_SIZE = 64
# Shuffle the training data
train_dataset = train_data.shuffle(BUFFER_SIZE)

# Batch and pad the datasets to the maximum length of the sequences
train_dataset = train_dataset.padded_batch(BATCH_SIZE)
test_dataset = test_data.padded_batch(BATCH_SIZE)

sample_string = 'Tensorflow, from basic to master!'
tokenized_string = tokenizer_subword.encode(sample_string)
print(f'tokenized string is: {tokenized_string}')
for str in tokenized_string:
    print(f'{str} -> {tokenizer_subword.decode([str])}')
print(f'original string is: {tokenizer_subword.decode(tokenized_string)}')

embedding_len = 64

model = keras.Sequential([
    keras.layers.Embedding(tokenizer_subword.vocab_size, embedding_len),
    keras.layers.GlobalAveragePooling1D(),
    keras.layers.Dense(6, activation=keras.activations.relu),
    keras.layers.Dense(1, activation=keras.activations.sigmoid)
])
print(model.summary())

model.compile(loss=keras.losses.binary_crossentropy, optimizer=keras.optimizers.Adam(), metrics=['accuracy'])

history = model.fit(train_dataset, epochs=10, validation_data=test_dataset)

show_model_history(history)

# Sample movie reviews
positive_review = "I absolutely loved this movie! The acting was superb, and the storyline kept me engaged throughout."
negative_review = "Unfortunately, this movie was a disappointment. The plot was confusing, and the characters lacked depth."
# Tokenize and encode the sample reviews
positive_tokens = tokenizer_subword.encode(positive_review)
negative_tokens = tokenizer_subword.encode(negative_review)
# Predict the sentiment using the trained model
positive_prediction = model.predict(pad_sequences([positive_tokens], maxlen=model.layers[0].output_shape[1]))
negative_prediction = model.predict(pad_sequences([negative_tokens], maxlen=model.layers[0].output_shape[1]))
# Display the model predictions
print("Model Prediction for Positive Review:", positive_prediction[0][0])
print("Model Prediction for Negative Review:", negative_prediction[0][0])
