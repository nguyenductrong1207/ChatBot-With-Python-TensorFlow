import os
from django.conf import settings
from django.shortcuts import render
from django.http import JsonResponse
import random
import json
import pickle
import numpy as np
import nltk
from nltk.stem import WordNetLemmatizer
from keras.models import load_model

lemmatizer = WordNetLemmatizer()

# Use os.path.join to create the path to the files
intents_path = os.path.join(settings.BASE_DIR, 'chatbot/models/intents.json')
model_path = os.path.join(settings.BASE_DIR, 'chatbot/models/chatbot_model.h5')
words_path = os.path.join(settings.BASE_DIR, 'chatbot/models/words.pkl')
classes_path = os.path.join(settings.BASE_DIR, 'chatbot/models/classes.pkl')

with open(intents_path, 'r') as file:
    intents = json.load(file)

words = pickle.load(open(words_path, 'rb'))
classes = pickle.load(open(classes_path, 'rb'))
model = load_model(model_path)

def clean_up_sentence(sentence):
    sentence_words = nltk.word_tokenize(sentence)
    sentence_words = [lemmatizer.lemmatize(word) for word in sentence_words]
    return sentence_words

def bag_of_words(sentence):
    sentence_words = clean_up_sentence(sentence)
    bag = [0] * len(words)
    for w in sentence_words:
        for i, word in enumerate(words):
            if word == w:
                bag[i] = 1
    return np.array(bag)

def predict_class(sentence):
    bow = bag_of_words(sentence)
    res = model.predict(np.array([bow]))[0]
    ERROR_THRESHOLD = 0.25
    results = [[i, r] for i, r in enumerate(res) if r > ERROR_THRESHOLD]
    results.sort(key=lambda x: x[1], reverse=True)
    return_list = []
    for r in results:
        return_list.append({'intent': classes[r[0]], 'probability': str(r[1])})
    return return_list

def get_response(intents_list, intents_json):
    tag = intents_list[0]['intent']
    list_of_intents = intents_json['intents']
    for i in list_of_intents:
        if i['tag'] == tag:
            result = random.choice(i['responses'])
            break
    return result

def home(request):
    return render(request, 'chatbot/index.html')

def chat(request):
    message = json.loads(request.body)['message']
    ints = predict_class(message)
    res = get_response(ints, intents)
    return JsonResponse({'response': res})
