import random as dom
import json
import pickle as pic
import numpy as np
import nltk
import tensorflow as tf
from nltk.stem import WordNetLemmatizer

lamatizer = WordNetLemmatizer()

intents = json.loads(open('intents.json').read())
words = pic.load(open('words.pkl', 'rb'))
classes = pic.load(open('classes.pkl', 'rb'))
model = tf.keras.models.load_model('chatbot_model.model')


def clean_up(sentence):
    sentence_words = nltk.word_tokenize(sentence)
    sentence_words = [lamatizer.lemmatize(word) for word in sentence_words]
    return sentence_words


def bag_words(sentence):
    sentence_words = clean_up(sentence)
    bag = [0] * len(words)
    for w in sentence_words:
        for i, word in enumerate(words):
            if word == w:
                bag[i] = 1
    return np.array(bag)


def predict(sentence):
    bow = bag_words(sentence)
    res = model.predict(np.array([bow]))[0]
    ERROR_THRESHOLD = 0.1
    results = [[i, r] for i, r in enumerate(res) if r > ERROR_THRESHOLD]

    results.sort(key=lambda x: x[1], reverse=True)
    return_list = []
    for r in results:
        return_list.append({'intent': classes[r[0]], 'probability': str(r[1])})
    return return_list


def _get_response(ints, intents_json):
    try:
        tag = ints[0]['intent']
        list_of_intents = intents_json['intents']
        for i in list_of_intents:
            if i['tag'] == tag:
                result = dom.choice(i['responses'])
                break
    except IndexError:
        result = "I don't understand!"
    return result


print("bot is running")

while True:
    message = input("")
    ints = predict(message)
    res = _get_response(ints, intents)
    print(res)
