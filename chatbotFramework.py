#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import sys
import random

import nltk
from nltk.stem.lancaster import LancasterStemmer
stemmer = LancasterStemmer()

import numpy as np
import tflearn
import tensorflow as tf

import pickle

import pymongo
from pymongo import MongoClient

# tokenize sentence and stem the words
def clean_up_sentence(sentence):
    sentence_words = nltk.word_tokenize(sentence)
    sentence_words = [stemmer.stem(word.lower()) for word in sentence_words]
    return sentence_words

# return bag of words array: 0 or 1 for each word in the bag that 
# exists in the sentence
def bow(sentence, words, debug=False):
    sentence_words = clean_up_sentence(sentence)
    # bag of words
    bag = [0]*len(words)  
    for s in sentence_words:
        for i, w in enumerate(words):
            if w == s: 
                bag[i] = 1
                if debug:
                    print ("found in bag: %s" % w)

    return (np.array(bag))

def classify(sentence):
    # get classification probabilities
    results = model.predict([bow(sentence, words)])[0]
    # remove predictions below the threshold
    results = [[i,r] for i,r in enumerate(results) if r>ERROR_THRESHOLD]
    
    # sort by probability
    results.sort(key=lambda x: x[1], reverse=True)
    return_list = []
    for r in results:
        return_list.append((classes[r[0]], r[1]))
        
    # return intent and probability tuple
    return return_list


ERROR_THRESHOLD = 0.25
context = {}


# always the same user for now
def response(sentence, userID='nerd', debug=True):
    results = classify(sentence)
    if debug: print(results)
    
    if results:
        if userID in context:
            doc = db.ai_intents.find_one({'$and':[{'name': results[0][0]},{'contextFilter' : { '$exists': True, '$eq': context[userID] }}]})
            del context[userID]
            if debug: print(context)
                
        doc = db.ai_intents.find_one({'name': results[0][0]})
        if 'contextSet' in doc and doc['contextSet']:
            if debug: print('contextSet=', doc['contextSet'])
            context[userID] = doc['contextSet']
            
        return print(random.choice(doc['responses']))

    else:
        print('I dont know what we are talking about.')
        
def load_model():
    # restore data structures
    data = pickle.load( open( "training_data", "rb" ) )

    global words
    global classes 
    
    words = data['words']
    classes = data['classes']
    train_x = data['train_x']
    train_y = data['train_y']
    
    # reset underlying graph data
    tf.reset_default_graph()
    
    # Build a neural network
    net = tflearn.input_data(shape=[None, len(train_x[0])])
    net = tflearn.fully_connected(net, 8)
    net = tflearn.fully_connected(net, 8)
    net = tflearn.fully_connected(net, len(train_y[0]), activation='softmax')
    net = tflearn.regression(net)
       
    # load saved model
    model = tflearn.DNN(net, tensorboard_dir='tflearn_logs')
    model.load('./model.tflearn')
    return model
            
def prompt_user():
    print ('Type "quit" to exit.')
    while (True):
        line = input('enter> ')
        if line == 'quit':
            sys.exit()
        response(line, debug=True)

       
# connect to mongodb and set the Intents database for use
client = MongoClient('mongodb://localhost:27017/')
db = client.Intents

model = load_model()
prompt_user()



