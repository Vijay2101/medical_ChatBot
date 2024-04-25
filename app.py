import streamlit as st
from streamlit import session_state
import tensorflow as tf
import pickle
import numpy as np
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from sklearn.preprocessing import LabelEncoder
import random
import nltk
from nltk.tokenize import word_tokenize
import string
import json
import random

hide_menu_style = """
        <style>
        #MainMenu {visibility: hidden;}
            footer {visibility: hidden;}
        </style>
        """
st.markdown(hide_menu_style, unsafe_allow_html=True)



with open("medical_intents.json") as data:
  dataset = json.load(data)

def processing_json_dataset(dataset):
  tags = []
  inputs = []
  responses={}
  for intent in dataset['intents']:
    responses[intent['tag']]=intent['responses']
    for lines in intent['patterns']:
      inputs.append(lines)
      tags.append(intent['tag'])
  return [tags, inputs, responses]

[tags, inputs, responses] = processing_json_dataset(dataset)

import keras.models
import pickle

model2 = keras.models.load_model('medi_model.h5')

# loading
with open('tokenizer.pkl', 'rb') as handle:
    tokenizer = pickle.load(handle)

with open('labelEncoder.pkl', 'rb') as handle:
    labelEncoder = pickle.load(handle)





def generate_answer(query):
  texts = []
  pred_input = query
  pred_input = [letters.lower() for letters in pred_input if letters not in string.punctuation]
  pred_input = ''.join(pred_input)
  texts.append(pred_input)
  pred_input = tokenizer.texts_to_sequences(texts)
  pred_input = np.array(pred_input).reshape(-1)
  pred_input = pad_sequences([pred_input],11)
  output = model2.predict(pred_input)
  output = output.argmax()
  print(output)
  response_tag = labelEncoder.inverse_transform([output])[0]
  return random.choice(responses[response_tag])


st.title("Medical ChatBot")

if "messages" not in st.session_state:
    st.session_state.messages = []

for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])
        
suggestion1 = "I have fever"
submit1 = st.button(suggestion1) 

suggestion2 = "How to cure headache"
submit2 = st.button(suggestion2) 

suggestion3 = "How do you treat a sprain?"
submit3 = st.button(suggestion3) 
       


if prompt := st.chat_input("What is up?"):
    st.session_state.messages.append({"role": "user", "content": prompt})
    
    
    with st.chat_message("user"):
        st.markdown(prompt)

    with st.chat_message("assistant"):
        ans = generate_answer(prompt)
        print(ans)  
        st.markdown(ans)
        st.session_state.messages.append({"role": "assistant", "content": ans})
        # st.session_state.medical_model.append({"role": "assistant", "content": ans})
        
if submit1:
  st.session_state.messages.append({"role": "user", "content": suggestion1})
  
  
  with st.chat_message("user"):
      st.markdown(suggestion1)

  with st.chat_message("assistant"):
      ans = generate_answer(suggestion1)
      print(ans)  
      st.markdown(ans)
      st.session_state.messages.append({"role": "assistant", "content": ans})
        
if submit2:
  st.session_state.messages.append({"role": "user", "content": suggestion2})
  
  
  with st.chat_message("user"):
      st.markdown(suggestion2)

  with st.chat_message("assistant"):
      ans = generate_answer(suggestion2)
      print(ans)  
      st.markdown(ans)
      st.session_state.messages.append({"role": "assistant", "content": ans})

if submit3:
  st.session_state.messages.append({"role": "user", "content": suggestion3})
  
  
  with st.chat_message("user"):
      st.markdown(suggestion3)

  with st.chat_message("assistant"):
      ans = generate_answer(suggestion3)
      print(ans)  
      st.markdown(ans)
      st.session_state.messages.append({"role": "assistant", "content": ans})




