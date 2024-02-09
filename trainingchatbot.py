import random
import json
import pickle
import numpy as np

import nltk
from nltk.stem import WordNetLemmatizer #reduce words to its stem so that we don't lose performance (it will looking for the exact words) ex: works, worked, working = work
import nltk
nltk.download('punkt')
import nltk
nltk.download('wordnet')

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Activation, Dropout
from tensorflow.keras.optimizers import SGD

lemmatizer = WordNetLemmatizer() #lemmatize individual words

intents = json.loads(open('C:/Users/PC/Desktop/FYP/FYP2/Updated/System Yasmin/chatborv2/intents.json').read()) #reading contents on json file (the dictionary/dataset)
#^ the object from our dataset/dictionary

#load training data
words = []
classes = []
documents = []
ignore_letters = ['?','!','.',',']

for intent in intents['intents']:
    for pattern in intent['patterns']:
        word_list = nltk.word_tokenize(pattern) #split into individual words
        words.extend(word_list) #masuk dlm word list
        documents.append((word_list, intent['tag'])) #word list belong to 'tag' category
        if intent['tag'] not in classes: #check if the class already in the list
            classes.append(intent['tag'])
#print(documents)

#prepare training data
words = [lemmatizer.lemmatize(word) for word in words if word not in ignore_letters]
words = sorted(set(words))

classes = sorted(set(classes))

pickle.dump(words, open('words.pkl','wb')) #save into file
pickle.dump(classes, open('classes.pkl','wb'))
#print(words)

#ml part. since RNN cannot be feed with words - change to number
training = []
output_empty = [0] * len(classes)

for document in documents:
    bag = []
    word_patterns = document[0]
    word_patterns = [lemmatizer.lemmatize(word.lower()) for word in word_patterns]
    for word in words: #check if words occurs in patterns in intents.json
        bag.append(1) if word in word_patterns else bag.append(0)

    output_row = list(output_empty)
    output_row[classes.index(document[1])] = 1
    training.append([bag, output_row])

#preprocess - shuffle data
random.shuffle(training)
training = np.array(training)

train_x = list(training[:, 0])
train_y = list(training[:, 1]) # label and features to train RNN

#RNN model
model = Sequential()
model.add(Dense(128, input_shape =(len(train_x[0]),), activation='relu')) #input layer (Dense) w/ activation f(x) rectified linear unit 'relu'
model.add(Dropout(0.5))
model.add(Dense(64, activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(len(train_y[0]), activation='softmax')) #softmax f(x) = allow to add/scales/sum up results/label in output layer (sort of % how likely to have the output)

#define stochastic gradient descent optimizer
sgd = SGD(lr=0.01, decay=1e-6, momentum=0.9, nesterov=True) #lr = learning rate
model.compile(loss='categorical_crossentropy', optimizer=sgd, metrics=['accuracy']) #compile

hist = model.fit(np.array(train_x), np.array(train_y), epochs=200, batch_size=5, verbose=1)
model.save('chatbotmodel.h5', hist)
print("Done. RNN is now trained :3")