# Step 1: Import Libraries and Load the Model
import os

import keras
from keras.src.datasets import imdb
from keras.src.utils import pad_sequences

# Load the IMDB dataset word index
word_index = imdb.get_word_index()
reverse_word_index = {value: key for key, value in word_index.items()}

# Load the pre-trained model with ReLU activation

# Define the project's root directory
project_root = "/Users/sunnythesage/PythonProjects/Data-Science-BootCamp/03-Deep-Learning-BootCamp/9 - End to End Deep Learning Project with Simple RNN/advanced-movie-review-sentiment-classification-using-simple-RNN"

# Change the current working directory to the project's root
os.chdir(project_root)

# --- Artifacts ---

# Define the relative path to the artifacts directory
artifacts_dir = os.path.join(os.getcwd(), "artifacts")

# Create the directory if it doesn't exist
os.makedirs(artifacts_dir, exist_ok=True)

# Load the pre-trained model with ReLU activation
model = keras.saving.load_model(os.path.join(artifacts_dir, "simple_rnn_model.keras"))


# Step 2: Helper Functions
# Function to decode reviews
def decode_review(encoded_review):
    return " ".join([reverse_word_index.get(i - 3, "?") for i in encoded_review])


# Function to preprocess user input
def preprocess_text(text):
    words = text.lower().split()
    encoded_review = [word_index.get(word, 2) + 3 for word in words]
    padded_review = pad_sequences([encoded_review], maxlen=500)
    return padded_review


import streamlit as st

## streamlit app
# Streamlit app
st.title("IMDB Movie Review Sentiment Analysis")
st.write("Enter a movie review to classify it as positive or negative.")

# User input
user_input = st.text_area("Movie Review")

if st.button("Classify"):

    preprocessed_input = preprocess_text(user_input)

    ## MAke prediction
    prediction = model.predict(preprocessed_input)
    sentiment = "Positive" if prediction[0][0] > 0.5 else "Negative"

    # Display the result
    st.write(f"Sentiment: {sentiment}")
    st.write(f"Prediction Score: {prediction[0][0]}")
else:
    st.write("Please enter a movie review.")
