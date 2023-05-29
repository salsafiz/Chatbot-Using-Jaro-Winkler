import string
import pickle
import numpy as np
import nltk
from nltk.stem import WordNetLemmatizer
from nltk.corpus import stopwords
import jellyfish

import json
import random
from flask import Flask, render_template, request

nltk.download('popular')
nltk.download('punkt')
nltk.download('stopwords')

app = Flask(__name__)
app.static_folder = 'static'

lemmatizer = WordNetLemmatizer()
intents = json.loads(open('maneh.json').read())

def clean_up_sentence(sentence):
    lemmatizer = WordNetLemmatizer()
    stop_words = set(stopwords.words('indonesian'))  # Menggunakan stopwords dalam bahasa Indonesia

    sentence_words = nltk.word_tokenize(sentence)
    sentence_words = [lemmatizer.lemmatize(word.lower()) for word in sentence_words if word.lower() not in stop_words]
    
    return sentence_words

def get_jw_similarity(input_str, tag_str):
    return jellyfish.jaro_winkler(input_str, tag_str)

def bow(sentence, words, show_details=True):
    # Tokenisasi pola
    sentence_words = clean_up_sentence(sentence)
    # Bag of words - matriks N kata, matriks kosakata
    bag = [0] * len(words)
    for s in sentence_words:
        for i, w in enumerate(words):
            if w == s:
                # Mengisi 1 jika kata saat ini ada di posisi kosakata
                bag[i] = 1
                if show_details:
                    print("Ditemukan dalam bag: %s" % w)
    return np.array(bag)

def predict_class(sentence, intents):
    p = bow(sentence, words, show_details=False)
    res = []
    for intent in intents['intents']:
        tag = intent['tag']
        patterns = intent['patterns']
        for pattern in patterns:
            sim = get_jw_similarity(sentence, pattern)
            res.append((tag, sim))
    res.sort(key=lambda x: x[1], reverse=True)
    return res

def get_response(tag, intents):
    list_of_intents = intents['intents']
    for intent in list_of_intents:
        if intent['tag'] == tag:
            responses = intent['responses']
            return random.choice(responses)
    return "Pertanyaan anda akan dijawab owner pada jam kerja"

def chatbot_response(msg):
    pred_tags = [tag for tag, sim in predict_class(msg, intents)]
    if pred_tags:
        tag = pred_tags[0]
        sim = predict_class(msg, intents)[0][1]
        res = get_response(tag, intents)
        if sim < 0.8:
            res = "Pertanyaan anda tidak sesuai"
    else:
        res = "Pertanyaan anda tidak sesuai"
    return res

@app.route("/")
def home():
    return render_template("index.html")

@app.route("/get")
def get_bot_response():
    user_text = request.args.get('msg')
    return chatbot_response(user_text)

if __name__ == "__main__":
    words = set()
    for intent in intents['intents']:
        patterns = intent['patterns']
        for pattern in patterns:
            sentence_words = clean_up_sentence(pattern)
            words.update(sentence_words)
    words = sorted(list(words))
    app.run()
