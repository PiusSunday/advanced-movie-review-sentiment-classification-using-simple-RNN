# Advanced Movie Review Sentiment Classification Using Simple RNN

This project focuses on building a sentiment classification model using a Simple Recurrent Neural Network (RNN) to
analyze movie reviews from the IMDB dataset.
The goal is to accurately predict the sentiment (positive or negative)
expressed in each review.
The project encompasses data preprocessing, RNN model development, evaluation, and deployment
using Streamlit for interactive testing.

## Problem Statement:

Sentiment analysis of movie reviews is crucial for understanding audience perception and feedback.
This project aims to
develop a robust sentiment classification model using a Simple RNN to accurately predict the sentiment (positive or
negative) of movie reviews from the IMDB dataset.
By accurately classifying reviews, we can provide valuable insights
into audience reactions, aiding in content creation and marketing strategies.

## Motivation and Business Value:

Understanding audience sentiment is vital for the entertainment industry.
Accurate sentiment analysis can inform
decisions related to movie marketing, content development, and audience engagement.
This project demonstrates the
application of deep learning techniques, specifically Simple RNNs, to effectively analyze textual data and extract
meaningful sentiment insights.
Simple RNNs are particularly useful for sequential data like a text, allowing us to
capture
the context and dependencies within reviews.

## Dataset:

The project uses the IMDB dataset, a widely recognized benchmark dataset for sentiment analysis.
It consists of
50,000 movie reviews, equally split into 25,000 training and 25,000 testing samples.
Each review is labeled as either
positive or negative.
For more information, please refer to
the [IMDB dataset documentation](https://www.kaggle.com/datasets/lakshmi25npathi/imdb-dataset-of-50k-movie-reviews).

## Methodology and Implementation:

This project provides a comprehensive, reproducible workflow for sentiment analysis using Python within a Jupyter
Notebook environment.
The methodology encompasses:

1. **Data Ingestion and Preprocessing:**
    * Loading the IMDB dataset.
    * Text cleaning, including removal of HTML tags, punctuation, and special characters.
    * Tokenization and vocabulary creation.
    * Padding sequences to ensure uniform input length for the RNN.
    * Conversion of text data into numerical sequences suitable for RNN input.
2. **Exploratory Data Analysis (EDA):**
    * Analyzing the distribution of positive and negative reviews.
    * Examining the length distribution of reviews.
    * Visualizing word frequency and common terms.
3. **Simple RNN Model Development:**
    * Building a Simple RNN architecture using Keras/TensorFlow.
    * Embedding layer for word vector representation.
    * RNN layer to capture sequential dependencies.
    * Dense layers for classification.
    * Hyperparameter tuning and optimization.
4. **Model Evaluation and Validation:**
    * Splitting the dataset into training, validation, and testing sets.
    * Evaluating model performance using metrics such as accuracy, precision, recall, F1-score, and AUC-ROC.
    * Visualizing training and validation loss/accuracy curves.
5. **Model Deployment and Streamlit Integration:**
    * Saving the trained model.
    * Developing a Streamlit application (`main.py`) for interactive testing.
    * Creating a user-friendly interface for inputting movie reviews and displaying sentiment predictions.

## Key Technical Aspects:

* Utilization of TensorFlow/Keras for deep learning model development.
* Implementation of word embeddings for effective text representation.
* Handling sequential data using Simple RNNs.
* Streamlit for interactive model deployment and testing.

## Streamlit Integration:

This project includes a Streamlit application (`main.py`)
for easy deployment and interaction with the trained Simple RNN
model.
Users can input movie reviews through a user-friendly interface, and the application will predict the sentiment
of the review in real-time.

To run the Streamlit app:

1. Ensure you have Streamlit installed (`pip install streamlit`).
2. Navigate to the project's root directory in your terminal.
3. Run the command `streamlit run app.py`.

This will launch the application in your web browser, allowing you to test the model with various movie review inputs.

## Potential Applications:

* **Movie Industry Feedback Analysis:** Understanding audience sentiment towards newly released movies.
* **Content Recommendation Systems:** Filtering and recommending movies based on sentiment.
* **Social Media Monitoring:** Analyzing sentiment in movie-related social media posts.
* **Market Research:** Gathering insights into audience preferences and trends.

## Note on Model Selection:

This project leverages a Simple RNN due to its ability to handle sequential data and capture dependencies in text.
While
more advanced RNN architectures like LSTM and GRU could be considered for enhanced performance, this project focuses on
demonstrating the fundamental principles of RNN-based sentiment classification with a simple and effective model.