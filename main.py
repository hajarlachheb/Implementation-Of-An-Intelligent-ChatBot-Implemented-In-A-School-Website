import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'#ignore warnings

import nltk
from nltk.stem import WordNetLemmatizer

lemmatizer = WordNetLemmatizer()
import pickle
import numpy as np

from keras.models import load_model

import json
import random

#load intents, tags, words and trained model
intents = json.loads(open('intents.json').read())
words = pickle.load(open('words.pkl', 'rb'))
tags = pickle.load(open('tags.pkl', 'rb'))
model = load_model('ChatBot_Trained_Model.h5')

def run():
    print("Welcome to ESI Chatbot.I will be assisting you today so that you get to know about ESI!")
    while (True):
        user_question = input("User : ")
        print("Chatbot is typing ...")
        chatbot_answer = predictAnswer(user_question)
        print("Chatbot: ", chatbot_answer[0])
    	#quit the while loop on goodbye tag
        try:
            if (chatbot_answer[1] != ""):
                break
        except:
            IndexError
def predictAnswer(question):
    intent = predictTag(question, model)#predict the tag for question with given trained model
    answer = getAnswer(intent, intents)#returns random answer given tag
    return answer

def clean_up_sentence(sentence):
    # tokenize the pattern - split words into array
    sentence_words = nltk.word_tokenize(sentence)
    # stem each word - create short form for word
    sentence_words = [lemmatizer.lemmatize(word.lower()) for word in sentence_words]
    return sentence_words


# return bag of words array: 0 or 1 for each word in the bag that exists in the sentence

def bow(sentence, words, show_details=True):
    # tokenize the pattern
    sentence_words = clean_up_sentence(sentence)
    # bag of words - matrix of N words, vocabulary matrix
    bag = [0] * len(words)
    for s in sentence_words:
        for i, w in enumerate(words):
            if w == s:
                # assign 1 if current word is in the vocabulary position
                bag[i] = 1
                if show_details:
                    print("found in bag: %s" % w)
    return (np.array(bag))


def predictTag(sentence, model):
    extractedWords = bow(sentence, words, show_details=False)#Bag of words to extract text from words
    answer = model.predict(np.array([extractedWords]))[0]
    ERROR_THRESHOLD = 0.25
    results = []
    for intent, error in enumerate(answer):
        if error > ERROR_THRESHOLD:
            results.append([intent, error])
    # sort by probability
    results.sort(key=lambda x: x[1], reverse=True)
    return_list = []
    for result in results:
        return_list.append({"intent": tags[result[0]], "probability": str(result[1])})#intent probability
    return return_list


def getAnswer(intent, intents_json):
    tag = intent[0]['intent']
    all_intents = intents_json['intents']
    #if tag is bye => quit
    result = []
    for intent in all_intents:
        if (intent['tag'] == tag):
            result.append(random.choice(intent['responses']))
            break
        if (tag == "goodbye"):
            result.append(tag)
    return result

run()



