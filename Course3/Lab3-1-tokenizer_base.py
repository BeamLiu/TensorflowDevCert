from keras.preprocessing.text import Tokenizer
from keras.src.utils import pad_sequences

sentences = [
    'I love my cat',
    'I love my dog',
    'You love my dog!',
    "Do you think my dog is amazing?"
]
tokenizer = Tokenizer(num_words=100, oov_token='<OOV>')
# token the input sentenes
tokenizer.fit_on_texts(sentences)
# Get the word index dictionary
word_index = tokenizer.word_index
# generate list of token sequences
sequenes = tokenizer.texts_to_sequences(sentences)
# pad the sequence
padded = pad_sequences(sequenes, padding='post', truncating='post', maxlen=5)

print(f'word index: {word_index}')
print((f'sequences: {sequenes}'))
print(f'padded sequence:\n {padded}')

test_data = [
    'i really love my dog',
    'my dog loves my manatee'
]
test_seq = tokenizer.texts_to_sequences(test_data)
test_padded = pad_sequences(test_seq, maxlen=10)
print(f'word index: {word_index}')
print((f'test sequences: {test_seq}'))
print(f'test padded sequence:\n {test_padded}')
