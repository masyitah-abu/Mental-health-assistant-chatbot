# libraries
import csv
import random
import numpy as np
import pickle
import json
from flask import Flask, render_template, request
import nltk
from tensorflow.keras.models import load_model
from nltk.stem import WordNetLemmatizer
lemmatizer = WordNetLemmatizer()
import json
from flask import Flask, jsonify, redirect, render_template, request,url_for
from tensorflow.keras.models import model_from_json

# chat initialization
model1 = load_model("C:/Users/HTC/Desktop/chatbotv3_2/chatbotmodel.h5")
intents = json.loads(open("C:/Users/HTC/Desktop/chatbotv3_2/intents.json").read())
words = pickle.load(open("C:/Users/HTC/Desktop/chatbotv3_2/words.pkl", "rb"))
classes = pickle.load(open("C:/Users/HTC/Desktop/chatbotv3_2/classes.pkl", "rb"))

import tensorflow_hub as hub
#mental health 
with open('C:/Users/HTC/Desktop/chatbotv3_2/model.json', 'r') as f: 
  json = f.read() 
model = model_from_json(json, custom_objects={'KerasLayer': hub.KerasLayer})

app = Flask(__name__)

#__________main page_______________
@app.route("/")
def home():
    return render_template("menu.html")#dh settle

#__________chatbot page_____________
@app.route("/chat",methods=["GET","POST"])
def chat():
    return render_template("index.html")#dh dpt reply tpi xleh

@app.route("/get", methods=["GET","POST"])
def chatbot_response():
    msg = request.form["msg"]
    if msg.startswith('Result mental health'):
        surveys = []
        file = open("C:/Users/HTC/Desktop/chatbotv3_2/survey.csv")
        for line in file:
            if line == "" or line == "":
                continue
            surveys.append(line.split(","))

        mentalhealt=surveys[0][3:9]
        print(mentalhealt)
        results = model.predict(mentalhealt)

        preds=[]
        for i in range(len(results)):
            if results[i]>=0:
                A=1
            else:
                A=0
            preds.append(A)

        print(preds)
        x = preds.count(1)
        y = preds.count(0)

        happy = x/len(preds)*100
        sad = y/len(preds)*100
        print(happy,sad)
        
        if sad > happy:
            res="You are going through a bad phase in life. But don't worry, bad times are not permanent."
        else:
            res="Your mental health looks great!!!"
            
    elif msg.startswith('Result stress level'):
        surveys = []
        file = open("C:/Users/HTC/Desktop/chatbotv3_2/survey.csv")
        for line in file:
            if line == "" or line == "":
                continue
            surveys.append(line.split(","))
            score = []
            for i in range(9,19):
                if int(surveys[0][i]) == 1:
                    score.append(4)
                elif int(surveys[0][i]) == 2:
                    score.append(3)
                elif int(surveys[0][i]) == 3:
                    score.append(2)
                elif int(surveys[0][i]) == 4:
                    score.append(1)
                elif int(surveys[0][i]) == 5:
                    score.append(0)
                else:
                    score.append(0)

                stresslevel = sum(score)
   
                if stresslevel < 13:
                    res="You have low stress level, Score below 13"
                elif stresslevel > 14 & stresslevel < 26:
                    res="Your stress level is moderate, Score between 14 and 26"
                else :
                    res="Your stress level is high. Score above 26."
    elif msg.startswith('Test Form'):
        return my_form()
    else:
        ints = predict_class(msg, model1)
        res = getResponse(ints, intents)
    return res

# chat functionalities
def clean_up_sentence(sentence):
    sentence_words = nltk.word_tokenize(sentence)
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
    return np.array(bag)

def predict_class(sentence, model):
    # filter out predictions below a threshold
    p = bow(sentence, words, show_details=False)
    res = model.predict(np.array([p]))[0]
    ERROR_THRESHOLD = 0.25
    results = [[i, r] for i, r in enumerate(res) if r > ERROR_THRESHOLD]
    # sort by strength of probability
    results.sort(key=lambda x: x[1], reverse=True)
    return_list = []
    for r in results:
        return_list.append({"intent": classes[r[0]], "probability": str(r[1])})
    return return_list

def getResponse(ints, intents_json):
    tag = ints[0]["intent"]
    list_of_intents = intents_json["intents"]
    for i in list_of_intents:
        if i["tag"] == tag:
            result = random.choice(i["responses"])
            break
    return result

#_______form & sheet____________ #dh pi kat form dlm sheet pon dh save elok
@app.route('/my_form', methods=['GET','POST'])
def my_form():
    return render_template('form.html')

@app.route('/my_form_post', methods=['GET','POST'])
def my_form_post():
    name = request.form.get("name-input")
    age = request.form.get("age-input")
    gender = request.form.get("gender-input")
    MQ1 = request.form.get("MQ1-input")
    MQ2 = request.form.get("MQ2-input")
    MQ3 = request.form.get("MQ3-input")
    MQ4 = request.form.get("MQ4-input")
    MQ5 = request.form.get("MQ5-input")
    MQ6 = request.form.get("MQ6-input")
    SQ1 = request.form.get("SQ1-input")
    SQ2 = request.form.get("SQ2-input")
    SQ3 = request.form.get("SQ3-input")
    SQ4 = request.form.get("SQ4-input")
    SQ5 = request.form.get("SQ5-input")
    SQ6 = request.form.get("SQ6-input")
    SQ7 = request.form.get("SQ7-input")
    SQ8 = request.form.get("SQ8-input")
    SQ9 = request.form.get("SQ9-input")
    SQ10 = request.form.get("SQ10-input")

    csvF = open("C:/Users/HTC/Desktop/chatbotv3_2/survey.csv", "w",newline='')
    writer = csv.writer(csvF)
    writer.writerow([name, age, gender, MQ1,MQ2,MQ3,MQ4,MQ5,MQ6,SQ1,SQ2,SQ3,SQ4,SQ5,SQ6,SQ7,SQ8,SQ9,SQ10])
    csvF.close()

    return tableAppend("success")

@app.route("/sheet", methods=["GET"])
def get_sheet():
    return tableAppend("")

def tableAppend(success):
    surveys = []
    file = open("C:/Users/HTC/Desktop/chatbotv3_2/survey.csv")
    for line in file:
        if line == "" or line == "":
            continue
        surveys.append(line.split(","))
    return render_template("sheet.html", surveys=surveys,message=success)

#__________helper page______________#dh settle
@app.route("/helper",methods=["GET","POST"])
def helper():
    return render_template("healthcarehelper.html")

@app.route("/playlist",methods=["GET","POST"])
def playlist():
    return render_template("playlist.html")#tgk camna nk link kan music#done

@app.route("/heathy",methods=["GET","POST"])
def heathy():
    return render_template("healthyhabits.html")#camna nk link kan video#done

@app.route("/time",methods=["GET","POST"])
def time():
    return render_template("timemanagement.html")#camna nk link kan video

#___________contact page_______________#dh settle
@app.route("/contact",methods=["GET","POST"])
def contact():
    return render_template("contactpage.html")

if __name__ == "__main__":
    app.run()

