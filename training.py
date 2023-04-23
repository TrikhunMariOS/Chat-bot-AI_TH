import random
import json
import pickle
import numpy as np

from pythainlp.tokenize import word_tokenize

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Activation, Dropout
from tensorflow.keras.optimizers import SGD

#นำเข้าข้อมูลไฟล์จาก intents.json
with open('intents.json', encoding='utf-8') as f:
    intents = json.load(f)

words = []
classes = []
documents = []
ignore_letters = ['?','!','.',',','น่ะ','เอิ่ม','พอดี','แล้ว','อ่ะ','อะ',' ']

for intent in intents['intents']:
    for pattern in intent['patterns']:
        word_list = word_tokenize(pattern, keep_whitespace=False)
        words.extend(word_list)
        documents.append((word_list, intent['tag']))
        if intent['tag'] not in classes:
            classes.append(intent['tag'])

words = [word for word in words if word not in ignore_letters]
words = sorted(set(words))

classes = sorted(set(classes))

pickle.dump(words, open('words.pkl', 'wb'))
pickle.dump(classes, open('classes.pkl', 'wb'))

training = []
output_empty = [0] * len(classes)

for document in documents:
    bag = []
    word_patterns = document[0]
    for word in words:
        bag.append(1) if word in word_patterns else bag.append(0)

    output_row = list(output_empty)
    output_row[classes.index(document[1])] = 1
    training.append([bag,output_row])

random.shuffle(training)
training = np.array(training)

train_x = list(training[:,0])
train_y = list(training[:,1])

model = Sequential()
model.add(Dense(128, input_shape=(len(train_x[0]),), activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(64, activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(len(train_y[0]), activation='softmax'))

#การเทรนนิ่ง AI-----------------------------------------------------
from tensorflow.keras.optimizers.schedules import ExponentialDecay

initial_learning_rate = 0.01
decay_steps = 100000
decay_rate = 1e-6

lr_schedule = ExponentialDecay(initial_learning_rate, decay_steps, decay_rate, staircase=True)
sgd = SGD(learning_rate=lr_schedule, momentum=0.9, nesterov=True)

model.compile(loss='categorical_crossentropy', optimizer=sgd, metrics=['accuracy'])

#Early stopping เป็นเทคนิคที่ใช้ในการฝึกอบรมโมเดล deep learning เพื่อป้องกัน overfitting ,โดยจะหยุดการเทรนนิ่งเมื่อโมเดลไม่ไม่การปรับปรุงเพิ่มเติม
from tensorflow.keras.callbacks import EarlyStopping

early_stopping = EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)

hist = model.fit(np.array(train_x), np.array(train_y), epochs=500, batch_size=5, verbose=1, validation_split=0.2, callbacks=[early_stopping])
model.save('chatbot_model.model.h5',hist)
print('----------------','AI Training is done','----------------')
