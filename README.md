
# Medical Chatbot App

This is a simple chatbot application built using Streamlit and TensorFlow/Keras. It's designed to provide responses to medical-related queries. The chatbot is based on a pre-trained model that has been trained on a dataset of medical intents.


## Demo

You can also access a live demo of the Medical Chatbot App by clicking [here](https://med-chatbot.streamlit.app/).



## Features

- You can have a conversation with the chatbot by typing in your questions and queries.

- The chatbot will provide responses based on the medical intents it has been trained on.



## Getting Started

- Clone this repository to your local machine.

- Install the required packages using the following command:

```bash
  pip install streamlit tensorflow numpy nltk

```
- Download the pre-trained model files:
    - medi_model.h5
    - tokenizer.pkl
    - labelEncoder.pkl

- Download the dataset file:
    - medical_intents.json

- Place the downloaded files in the same directory as the app script.
    
## Usage


- Run the Streamlit app using the following command:
```bash
streamlit run your_app_file.py
```

- Open the app in your web browser.
- Type your medical-related questions and queries into the chat input box.
- The chatbot will provide responses based on the input and the pre-trained model.
- Access the chatbot application by clicking on the following link:

   (https://med-chatbot.streamlit.app/)


## Data and Model

- The chatbot uses a dataset stored in medical_intents.json to train and understand different medical intents.
- The pre-trained model is stored in medi_model.h5.
- The tokenizer.pkl and labelEncoder.pkl files are used to preprocess and post-process the text data.
## Tech Stack

- **Framework**: Streamlit
- **Machine Learning**: TensorFlow and Keras
- **Data Preprocessing**: Python Libraries (pickle, numpy, nltk, string)
- **Dataset**: Medical Intents JSON (from Kaggle)



## Credits

This app was created by Vijay Kumar. The medical intents dataset used in this application was sourced from Kaggle. I would like to express my gratitude to the Kaggle community for providing this valuable dataset. 

- Dataset Source: [Kaggle Medical Intents Dataset](https://www.kaggle.com/datasets/tusharkhete/dataset-for-medicalrelated-chatbots) by Tushar Khete
