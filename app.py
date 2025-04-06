import streamlit as st
import pickle
import numpy as np
import tensorflow as tf

model = tf.keras.models.load_model('translation_model.h5')
seq = tf.keras.preprocessing.sequence.pad_sequences(...)

# Load the model and tokenizers
with open('eng_tokenizer.pkl', 'rb') as f:
    eng_tokenizer = pickle.load(f)
with open('hin_tokenizer.pkl', 'rb') as f:
    hin_tokenizer = pickle.load(f)

# Get vocabulary size and max sequence length
max_len = 20  # Replace with the value you used during training

# Translation function
def translate(sentence):
    sentence = sentence.lower()
    seq1 = eng_tokenizer.texts_to_sequences([sentence])
    seq1 = seq(seq1, maxlen=max_len, padding='post')
    prediction = model.predict(seq)
    translated = []
    for i in range(max_len):
        word_index = np.argmax(prediction[0, i])
        if word_index > 0:
            word = hin_tokenizer.index_word.get(word_index, '')
            translated.append(word)
    return ' '.join(translated)

# Streamlit UI
st.set_page_config(page_title="English to Hindi Translator")
st.title("English to Hindi Translator")

user_input = st.text_input("Enter an English sentence:")
if st.button("Translate"):
    if user_input.strip() != "":
        result = translate(user_input)
        st.success(f"Hindi Translation: {result}")


