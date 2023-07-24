# -*- coding: utf-8 -*-
"""
Created on Thu Jun 22 20:59:00 2023

@author: Anuruddha
"""
import random
import json
import pickle
import numpy as np
import nltk

from nltk.stem import WordNetLemmatizer
from tensorflow.keras.models import load_model


lemmatizer = WordNetLemmatizer()

with open("intents.json") as file:
    intents = json.load(file)


# load saved things    
words = pickle.load(open("words.pkl","rb"))
classes = pickle.load(open("classes.pkl","rb"))
model = load_model("chatbot_model.h5")


# working functions
def clean_up_sentence(sentence):
    sentence_words = nltk.word_tokenize(sentence)
    sentence_words = [lemmatizer.lemmatize(word) for word in sentence_words]
    return sentence_words

def bag_of_words(sentence):
    sentence_words = clean_up_sentence(sentence)
    bag = [0]*len(words)
    for w in sentence_words:
        for i,word in enumerate(words):
            if word == w:
                bag[i]=1
    return np.array(bag)

def predict_class(sentence):
    bow = bag_of_words(sentence)
    res = model.predict(np.array([bow]))[0] # give numerical class probability array
    ERROR_THRESHOLD = 0.25
    results = [[i,r] for i,r in enumerate(res) if r > ERROR_THRESHOLD]
    
    results.sort(key = lambda x:x[1], reverse=True) # sort the reults by probability desc order
    
    return_list  =[]
    for r in results:
        return_list.append({'intent':classes[r[0]], 'probability':str(r[1])})
    return return_list 

def get_responce(intent_list,intents_json):
    tag = intent_list[0]['intent'] # most closing tag
    list_of_intents = intents_json['intents'] # get all dicts in json file
    for i in list_of_intents:
        if i['tag'] == tag:
            result = random.choice(i['responses'])
            break
    return result


# run the bot
print('GO! Bot is running!')

while True:
    message = input("")
    ints = predict_class(message)
    res = get_responce(ints,intents)
    print(res)
    if ints[0]['intent'] == 'goodbye':
        break








