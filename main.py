pip install jiwer

import collections
from collections import Counter
from keras.models import load_model

import helper
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
%matplotlib inline

from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from keras.models import Model, Sequential, load_model
from keras.layers import GRU, Input, Dense, TimeDistributed, Activation, RepeatVector, Bidirectional, LSTM, Embedding
from keras.optimizers.legacy import Adam
from keras.losses import sparse_categorical_crossentropy
from keras.callbacks import ModelCheckpoint

import plotly.express as px
from plotly.subplots import make_subplots
import plotly.graph_objects as go

from sklearn.model_selection import train_test_split

from tabulate import tabulate

import gc
import nltk
from nltk.translate.bleu_score import corpus_bleu
from nltk.translate.meteor_score import meteor_score
from jiwer import wer

# Load parallel corpus
def load_corpus(file_path, encoding='utf-8'):
    try:
        with open(file_path, 'r', encoding=encoding) as file:
            lines = file.read().splitlines()
        return lines
    except UnicodeDecodeError:
        print(f"Error decoding file: {file_path}. Trying with 'latin-1' encoding.")
        return load_corpus(file_path, encoding='latin-1')

# Replace with the actual paths to your downloaded files
romanian_corpus_path = '/content/RO-STS.train.ro'  # Adjust the file path
english_corpus_path = '/content/RO-STS.train.en'

romanian_corpus = load_corpus(romanian_corpus_path)
english_corpus = load_corpus(english_corpus_path)

# ...

# Tokenization and sequence padding for training
tokenizer_romanian = Tokenizer(filters='')  # Change the tokenizer name

# Replace the existing code for initializing tokenizer_english with this
tokenizer_english = Tokenizer(filters='', oov_token='<OOV>')
tokenizer_english.fit_on_texts(english_corpus)

romanian_sequences = tokenizer_romanian.texts_to_sequences(romanian_corpus)
english_sequences = tokenizer_english.texts_to_sequences(english_corpus)

# ...

# Pad sequences to the same length
max_length = 50  # Adjust as needed based on your data
padded_romanian_sequences = pad_sequences(romanian_sequences, maxlen=max_length, padding='post')
padded_english_sequences = pad_sequences(english_sequences, maxlen=max_length, padding='post')

# Ensure sequences are of the same length
min_len = min(len(padded_romanian_sequences), len(padded_english_sequences))
padded_romanian_sequences = padded_romanian_sequences[:min_len]
padded_english_sequences = padded_english_sequences[:min_len]

# Vocabulary size
romanian_vocab_size = len(tokenizer_romanian.word_index) + 1
english_vocab_size = len(tokenizer_english.word_index) + 1

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(
    padded_romanian_sequences, padded_english_sequences, test_size=0.1, random_state=42
)

# Define the model
def define_model(in_vocab, out_vocab, in_timesteps, out_timesteps, batch_size):
    model = Sequential()
    model.add(Embedding(in_vocab, batch_size, input_length=in_timesteps, mask_zero=True))
    model.add(LSTM(batch_size))
    model.add(RepeatVector(out_timesteps))
    model.add(LSTM(batch_size, return_sequences=True))
    model.add(Dense(out_vocab, activation='softmax'))
    return model

# Hyperparameters
batch_size = 64
learning_rate = 0.001
epochs = 20
validation_split = 0.1

# Create and compile the model
model = define_model(romanian_vocab_size, english_vocab_size, max_length, max_length, batch_size)
model.compile(loss='sparse_categorical_crossentropy', optimizer=Adam(learning_rate))

# Model checkpoint to save the best model
checkpoint_path = 'model.h1.MT'
checkpoint = ModelCheckpoint(checkpoint_path, monitor='val_loss', verbose=1, save_best_only=True, mode='min')

# Training the model
history = model.fit(X_train, y_train, epochs=epochs, batch_size=batch_size, validation_split=validation_split, callbacks=[checkpoint], verbose=1)

# Plot training history
plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.legend(['train', 'validation'])
plt.title("Train vs Validation - Loss")
plt.show()

# Evaluate the model on test data
test_loss = model.evaluate(X_test, y_test, verbose=0)
print(f'Test Loss: {test_loss:.4f}')

# Load the saved model
model = load_model(checkpoint_path)

# ...

# ...

# Example translation using beam search
def translate_sentence_beam_search(sentence, romanian_tokenizer, english_tokenizer, model, max_length, beam_width=3):
    sequence = romanian_tokenizer.texts_to_sequences([sentence])[0]
    sequence = pad_sequences([sequence], maxlen=max_length, padding='post')
    prediction = model.predict(sequence, verbose=0)[0]

    # Apply beam search
    translated_sentence = beam_search(prediction, english_tokenizer, beam_width)

    return translated_sentence

# ...

# Beam search function
def beam_search(predictions, tokenizer, beam_width):
    start_token_id = tokenizer.word_index.get('<start>')
    end_token_id = tokenizer.word_index.get('<end>')

    if start_token_id is None or end_token_id is None:
        raise ValueError("Start or end token not found in tokenizer word index.")

    # Rest of the code remains the same
    # ...



# Function to calculate Word Error Rate (WER)
def calculate_wer(reference, hypothesis):
    return wer(reference, hypothesis)

# Function to calculate Total Error Rate (TER)
def calculate_ter(reference, hypothesis):
    # You may need to preprocess your input to match the format expected by TER
    # See: http://www.cs.umd.edu/~snover/tercom/current/tercom_manual.pdf
    return ter(reference, hypothesis)

# Function to calculate METEOR score
def calculate_meteor(reference, hypothesis):
    return meteor_score([reference], hypothesis)

# Function to calculate BLEU score
def calculate_bleu(reference, hypothesis):
    references = [[reference.split()]]  # NLTK expects a list of references for each hypothesis
    hypothesis = hypothesis.split()

    return corpus_bleu(references, [hypothesis])

# User input loop
while True:
    # Get user input
    user_input = input("Enter a Romanian sentence (or 'exit' to quit): ")

    # Check if the user wants to exit
    if user_input.lower() == 'exit':
        break

    # Translate the user input
    translated_sentence = translate_sentence_beam_search(user_input, tokenizer_romanian, tokenizer_english, model, max_length)

    # Display the translation
    print(f'Romanian: {user_input}')
    print(f'English Translation: {translated_sentence}')

    # Evaluate translation using WER, TER, METEOR, BLEU
    wer_score = calculate_wer(user_input, translated_sentence)
    ter_score = calculate_ter(user_input, translated_sentence)
    meteor_score_value = calculate_meteor(user_input, translated_sentence)
    bleu_score = calculate_bleu(user_input, translated_sentence)

    print(f'WER: {wer_score}')
    print(f'TER: {ter_score}')
    print(f'METEOR: {meteor_score_value}')
    print(f'BLEU: {bleu_score}')
    print()
