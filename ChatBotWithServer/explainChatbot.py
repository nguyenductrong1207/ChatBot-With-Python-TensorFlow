import random  # Importing the random module for generating random choices
import json  # Importing the json module for handling JSON data
import pickle  # Importing the pickle module for serializing and deserializing Python objects
import numpy as np  # Importing numpy for numerical operations

import nltk  # Importing the Natural Language Toolkit for text processing
from nltk.stem import WordNetLemmatizer  # Importing the WordNetLemmatizer for lemmatizing words
from keras.models import load_model  # Importing load_model to load a pre-trained Keras model

lemmatizer = WordNetLemmatizer()  # Creating an instance of WordNetLemmatizer
intents = json.loads(open('D:\\Github\\ChatBot-With-Python-TensorFlow\\intents.json').read())  # Loading the intents JSON file

words = pickle.load(open('words.pkl', 'rb'))  # Loading the words list from a pickle file
classes = pickle.load(open('classes.pkl', 'rb'))  # Loading the classes list from a pickle file
model = load_model('chatbot_model.h5')  # Loading the pre-trained Keras model


def clean_up_sentence(sentence):
    # Tokenizing and lemmatizing the input sentence
    sentence_words = nltk.word_tokenize(sentence)  # Tokenizing the sentence into words
    sentence_words = [lemmatizer.lemmatize(word) for word in sentence_words]  # Lemmatizing each word
    return sentence_words  # Returning the cleaned-up sentence words

def bag_of_words(sentence):
    # Converting the sentence into a bag of words representation
    sentence_words = clean_up_sentence(sentence)  # Cleaning up the input sentence
    bag = [0] * len(words)  # Initializing a bag of words with zeroes
    for w in sentence_words:  # Iterating over each word in the cleaned-up sentence
        for i, word in enumerate(words):  # Iterating over the words list
            if word == w:  # If the word matches, set the corresponding bag position to 1
                bag[i] = 1
    return np.array(bag)  # Returning the bag of words as a numpy array

def predict_class(sentence):
    # Predicting the class of the input sentence
    bow = bag_of_words(sentence)  # Getting the bag of words representation
    res = model.predict(np.array([bow]))[0]  # Making a prediction using the model
    ERROR_THRESHOLD = 0.25  # Setting a threshold for filtering predictions
    results = [[i, r] for i, r in enumerate(res) if r > ERROR_THRESHOLD]  # Filtering predictions based on the threshold

    results.sort(key=lambda x: x[1], reverse=True)  # Sorting the results by probability in descending order
    return_list = []  # Initializing the return list
    for r in results:  # Iterating over the sorted results
        return_list.append({'intent': classes[r[0]], 'probability': str(r[1])})  # Appending the intent and probability
    return return_list  # Returning the list of predicted intents

def get_response(intents_list, intents_json):
    # Getting the response for the predicted intent
    tag = intents_list[0]['intent']  # Getting the first predicted intent
    list_of_intents = intents_json['intents']  # Getting the list of intents from the JSON file
    for i in list_of_intents:  # Iterating over each intent in the JSON file
        if i['tag'] == tag:  # If the intent tag matches the predicted intent
            result = random.choice(i['responses'])  # Randomly choosing a response from the intent's responses
            break  # Breaking the loop once a match is found
    return result  # Returning the chosen response

print("GO! Bot is running!")  # Printing a message indicating that the bot is running

while True:
    message = input("")  # Getting user input
    ints = predict_class(message)  # Predicting the class of the user input
    res = get_response(ints, intents)  # Getting the response for the predicted class
    print(res)  # Printing the response
