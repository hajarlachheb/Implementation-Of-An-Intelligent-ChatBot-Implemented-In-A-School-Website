#nltk.download('punkt') one time download
#nltk.download('wordnet') one time download
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'


# natural language toolkit
import nltk
#lemamatizer
from nltk.stem import WordNetLemmatizer
# for intents data
import json
#to dump processed data
import pickle
#to process data matrix (bags...)
import numpy as np
import warnings
warnings.filterwarnings("ignore", category=np.VisibleDeprecationWarning)
#to create model using the processed data
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Activation, Dropout
from tensorflow.keras.optimizers import SGD
#to shuffle data for training
import random


#initialize variables
words = [] #contains words from intents
tags = [] #tags of questions
words_tags = [] #combined
to_ignore = [',','?', '!']
#load data
data = open('intents.json').read()
intents = json.loads(data)
lemmatizer = WordNetLemmatizer()

for intent in intents['intents']:
    for pattern in intent['patterns']:
        #tokenize words/ get words from patern each one separated
        tmp_words = nltk.word_tokenize(pattern)
        words.extend(tmp_words) #extends the list with the parameter list
        # add documents in the corpus
        words_tags.append((tmp_words, intent['tag']))#each question tokenized + its label as an element of the list
        #get tag of this question
        tags.append(intent['tag'].lower())
tags = sorted(list(set(tags))) #filter tags to only unique ones as we don't need repetition during selection
#we do the same for words to remove duplicates for our data and we get their base form to cover maximum on questions
tmp_words = []
for word in words:
    if word not in to_ignore:
        base_form = lemmatizer.lemmatize(word.lower())#to remove duplicate set everything to lower also ignore ? !
        tmp_words.append(base_form)
tmp_words = sorted(list(set(tmp_words))) #remove dupes
words = tmp_words#all our words lemmatized

pickle.dump(words, open('words.pkl', 'wb'))#save our processed words for usage in another file that will use data
pickle.dump(tags, open('tags.pkl', 'wb'))#save tags


#train data
train = []
output_init = list([0] * len(tags))
for combination in words_tags:
    training_set = []#words set x y
    pattern_words = combination[0]#tokenized words
    #lemmatize the tokenized words to cover max of forms
    tmp_pattern_words = []
    for word in pattern_words:
        tmp_pattern_words.append(lemmatizer.lemmatize(word.lower()))
    pattern_words = tmp_pattern_words
    # training set...1 if word exist in pattern 0 otherwise
    for word in words:
        if word in pattern_words:
            training_set.append(1)
        else:
            training_set.append(0)

    #output... 1 if it is the tag of this pattern else 0
    #get index of the current tag in the tags array
    #but before that reinitialize our output to 0's
    output = output_init.copy()
    index = tags.index(combination[1])
    output[index] = 1
    #else keep it as a 0 = initialized value
    train.append([training_set, output])#pattern as 1's and 0's + its tag as 1's and 0's for training
# shuffle prepared training data for independent change on model
# dodges bias
random.shuffle(train)
train = np.array(train)
# create train data we give to x axis our patterns and to y their tags
train_x = list(train[:, 0])#patterns
train_y = list(train[:, 1])#tags
#we finished creating our training data

# Create model
# 3 layers. First layer 128 neurons, second layer 64 neurons
# and 3rd number of neurons = intents for logistic regression with softmax function
model = Sequential()#layers stacl
model.add(Dense(128, input_shape=(len(train_x[0]),), activation='relu'))#1st layer 128 neurons
model.add(Dropout(0.5))#break layer
model.add(Dense(64, activation='relu'))#2nd layer 64 neurons
model.add(Dropout(0.5))
model.add(Dense(len(train_y[0]), activation='softmax'))#3rd layer train data neurons

# Compile model
sgd = SGD(lr=0.01, decay=1e-6, momentum=0.9, nesterov=True)
model.compile(loss='categorical_crossentropy', optimizer=sgd, metrics=['accuracy'])

#saving model for use in main file with 600 training to reach max accuracy 100%
model_fit = model.fit(np.array(train_x), np.array(train_y), epochs=600, batch_size=5, verbose=1)
model.save('ChatBot_Trained_Model.h5', model_fit)

