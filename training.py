import tensorflow as tf
import random as dom
import json
import pickle as pic
import numpy as np
import nltk
# nltk.download('omw-1.4')
from nltk.stem import WordNetLemmatizer

lamatizer = WordNetLemmatizer()

intents = json.loads(open('intents.json').read())

words = []
classes = []
documents = []
ignore_letters = ['?', '!', '.', ',']

for intent in intents['intents']:
    for pattern in intent['patterns']:
        word_list = nltk.word_tokenize(pattern)
        words.extend(word_list)
        documents.append((word_list, intent['tag']))
        if intent['tag'] not in classes:
            classes.append(intent['tag'])


words = [lamatizer.lemmatize(word)
         for word in words if word not in ignore_letters]
words = sorted(set(words))
classes = sorted(set(classes))
pic.dump(words, open('words.pkl', 'wb'))
pic.dump(classes, open('classes.pkl', 'wb'))

training = []
output_empty = [0] * len(classes)

for document in documents:
    bag = []
    word_patterns = document[0]
    word_patterns = [lamatizer.lemmatize(
        word.lower()) for word in word_patterns]
    for word in words:
        bag.append(1) if word in word_patterns else bag.append(0)

        output_row = list(output_empty)
        output_row[classes.index(document[1])] = 1
        training.append([bag, output_row])

dom.shuffle(training)
training = np.array(training, dtype=object)

train_x = list(training[:, 0])
train_y = list(training[:, 1])

model = tf.keras.Sequential()
model.add(tf.keras.layers.Dense(
    128, input_shape=(len(train_x[0]),), activation='relu'))
model.add(tf.keras.layers.Dropout(0.5))
model.add(tf.keras.layers.Dense(64, activation='relu'))
model.add(tf.keras.layers.Dropout(0.5))
model.add(tf.keras.layers.Dense(len(train_y[0]), activation='softmax'))

adam = tf.keras.optimizers.legacy.SGD(
    learning_rate=0.01, decay=1e-6, momentum=0.9, nesterov=True)
model.compile(loss='categorical_crossentropy',
              optimizer='adam', metrics=['accuracy'])
model_hist = model.fit(np.array(train_x), np.array(train_y),
                       epochs=200, batch_size=5, verbose=1)
model.save('chatbot_model.model', model_hist)
print("done")
