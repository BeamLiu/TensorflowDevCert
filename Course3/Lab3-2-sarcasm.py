import json

from keras.src.preprocessing.text import Tokenizer
from keras.src.utils import pad_sequences

with open('./dataset/Sarcasm_Headlines_Dataset_v2.json', 'r') as file:
    datastore = json.load(file)

sentences = []
labels = []
urls = []

for item in datastore:
    sentences.append(item['headline'])
    labels.append(item['is_sarcastic'])
    urls.append(item['article_link'])

tokenizer = Tokenizer(oov_token='<OOV>')
# token the input sentenes
tokenizer.fit_on_texts(sentences)
# Get the word index dictionary
word_index = tokenizer.word_index
print(f'number of word index: {len(word_index)}')
print(f'word index: {word_index}')
# generate list of token sequences
sequenes = tokenizer.texts_to_sequences(sentences)
padded = pad_sequences(sequenes, padding='post')

index = 3
print(f'sample headline: {sentences[index]}')
print(f'padded sequence: {padded[index]}')
print(f'shape of padded sequence: {padded.shape}')
