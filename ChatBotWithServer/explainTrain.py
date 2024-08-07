import random  # Importing the random module for generating random choices
import json  # Importing the json module for handling JSON data
import pickle  # Importing the pickle module for serializing and deserializing Python objects
import numpy as np  # Importing numpy for numerical operations
import tensorflow as tf  # Importing TensorFlow for building and training the neural network

import nltk  # Importing the Natural Language Toolkit for text processing
from nltk.stem import WordNetLemmatizer  # Importing the WordNetLemmatizer for lemmatizing words

lemmatizer = WordNetLemmatizer()  # Creating an instance of WordNetLemmatizer

intents = json.loads(open('D:\\Github\\ChatBot-With-Python-TensorFlow\\intents.json').read())  # Loading the intents JSON file

words = []  # Initializing the words list
classes = []  # Initializing the classes list
documents = []  # Initializing the documents list
ignoreLetters = ['?', '!', '.', ',']  # List of characters to ignore

for intent in intents['intents']:  # Iterating over each intent in the intents JSON file
    for pattern in intent['patterns']:  # Iterating over each pattern in the intent
        wordList = nltk.word_tokenize(pattern)  # Tokenizing the pattern into words
        words.extend(wordList)  # Adding the tokenized words to the words list
        documents.append((wordList, intent['tag']))  # Appending the word list and intent tag to documents
        if intent['tag'] not in classes:  # Adding the intent tag to classes if it's not already present
            classes.append(intent['tag'])

words = [lemmatizer.lemmatize(word) for word in words if word not in ignoreLetters]  # Lemmatizing and filtering words
words = sorted(set(words))  # Sorting and removing duplicates from words

classes = sorted(set(classes))  # Sorting and removing duplicates from classes

pickle.dump(words, open('words.pkl', 'wb'))  # Saving the words list to a pickle file
pickle.dump(classes, open('classes.pkl', 'wb'))  # Saving the classes list to a pickle file

training = []  # Initializing the training data list
outputEmpty = [0] * len(classes)  # Creating an empty output list with the same length as classes

for document in documents:  # Iterating over each document
    bag = []  # Initializing the bag of words list
    wordPatterns = document[0]  # Getting the word patterns from the document
    wordPatterns = [lemmatizer.lemmatize(word.lower()) for word in wordPatterns]  # Lemmatizing the word patterns
    for word in words:  # Iterating over each word in words
        bag.append(1) if word in wordPatterns else bag.append(0)  # Adding 1 if the word is in word patterns, otherwise 0

    outputRow = list(outputEmpty)  # Creating a copy of the empty output list
    outputRow[classes.index(document[1])] = 1  # Setting the appropriate output index to 1
    training.append(bag + outputRow)  # Appending the bag of words and output row to training data

random.shuffle(training)  # Shuffling the training data
training = np.array(training)  # Converting the training data to a numpy array

trainX = training[:, :len(words)]  # Splitting the training data into input features
trainY = training[:, len(words):]  # Splitting the training data into output labels


model = tf.keras.Sequential()  # Creating a Sequential model
model.add(tf.keras.layers.Dense(128, input_shape=(len(trainX[0]),), activation='relu'))  # Adding a dense layer with 128 neurons
model.add(tf.keras.layers.Dropout(0.5))  # Adding a dropout layer with a 50% dropout rate
model.add(tf.keras.layers.Dense(64, activation='relu'))  # Adding another dense layer with 64 neurons
model.add(tf.keras.layers.Dropout(0.5))  # Adding another dropout layer with a 50% dropout rate
model.add(tf.keras.layers.Dense(len(trainY[0]), activation='softmax'))  # Adding a dense output layer with softmax activation

sgd = tf.keras.optimizers.SGD(learning_rate=0.01, momentum=0.9, nesterov=True)  # Creating an SGD optimizer
model.compile(loss='categorical_crossentropy', optimizer=sgd, metrics=['accuracy'])  # Compiling the model

hist = model.fit(np.array(trainX), np.array(trainY), epochs=200, batch_size=5, verbose=1)  # Training the model
model.save('chatbot_model.h5', hist)  # Saving the trained model
print('Done')  # Printing a message indicating that training is done
