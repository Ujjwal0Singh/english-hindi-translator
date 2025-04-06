import streamlit as st
import pickle
import tensorflow as tf
import numpy as np
from tensorflow.keras.preprocessing.sequence import pad_sequences

# Load model and tokenizers
model = tf.keras.models.load_model('translation_model.h5')

with open('eng_tokenizer.pkl', 'rb') as f:
    eng_tokenizer = pickle.load(f)
with open('hin_tokenizer.pkl', 'rb') as f:
    hin_tokenizer = pickle.load(f)

max_seq_len = 20  # Change to your actual max_eng_len

def preprocess_text(text):
    return text.lower()

def translate(sentence):
    sentence = preprocess_text(sentence)
    seq = eng_tokenizer.texts_to_sequences([sentence])
    seq = pad_sequences(seq, maxlen=max_seq_len, padding='post')
    prediction = model.predict(seq)

    translated_sentence = []
    for i in range(max_seq_len):
        word_index = np.argmax(prediction[0, i, :])
        if word_index > 0:
            word = hin_tokenizer.index_word.get(word_index, '')
            translated_sentence.append(word)
    return ' '.join(translated_sentence)

st.title("🗣️ English to Hindi Translator")
st.write("Type an English sentence and get a Hindi translation!")

input_text = st.text_input("Enter English sentence:")

if input_text:
    translation = translate(input_text)
    st.success(f"Hindi Translation: {translation}")
