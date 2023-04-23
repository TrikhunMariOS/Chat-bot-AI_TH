import random
import json
import pickle
import numpy as np
import datetime

from pythainlp.tokenize import word_tokenize
from tensorflow.keras.models import load_model

with open('D:\\pythonProject6\\Scripts\\intents.json', encoding='utf-8') as f:
    intents = json.load(f)

words = pickle.load(open('D:\\pythonProject6\\Scripts\\words.pkl', 'rb'))
classes = pickle.load(open('D:\\pythonProject6\\Scripts\\classes.pkl', 'rb'))
model = load_model('D:\\pythonProject6\\Scripts\\chatbot_model.model.h5')


def clean_up_sentence(sentence):
    sentence_words = word_tokenize(sentence, keep_whitespace=False)
    return sentence_words


def bag_of_words(sentence):
    sentence_words = clean_up_sentence(sentence)
    bag = [0] * len(words)
    for w in sentence_words:
        for i, word in enumerate(words):
            if word == w:
                bag[i] = 1
    return np.array(bag)

#ชุดข้อมูลตอบคำถามทั่วไป
def predict_class(sentence):
    bow = bag_of_words(sentence)
    res = model.predict(np.array([bow]))[0]
    ERROR_THRESHOLD = 0.25
    results = [[i, r] for i,r in enumerate(res) if r > ERROR_THRESHOLD]

    results.sort(key=lambda x: x[1], reverse=True)
    return_list = []
    for r in results:
        return_list.append({'intents':classes[r[0]], 'probability':str(r[1])})
    return return_list

#ชุดข้อมูลตอบคำถามพิเศษ"นำข้อมูลจากภายนอกมาใช้"
def get_current_time(): #เวลา
    current_time = datetime.datetime.now()
    formatted_time = current_time.strftime("%H:%M:%S")
    return f"ตอนนี้เป็นเวลา {formatted_time} ค่ะ"

def get_current_date(): #วัน/เดือน/ปี
    current_date = datetime.datetime.now()
    formatted_date = current_date.strftime("%Y-%m-%d")
    return f"วันนี้คือวันที่ {formatted_date} ค่ะ"

def get_current_day(): #วันอะไร
    current_day = datetime.datetime.now()
    day_of_week = current_day.strftime("%A")
    days_translation = {
        'Monday': 'วันจันทร์',
        'Tuesday': 'วันอังคาร',
        'Wednesday': 'วันพุธ',
        'Thursday': 'วันพฤหัสบดี',
        'Friday': 'วันศุกร์',
        'Saturday': 'วันเสาร์',
        'Sunday': 'วันอาทิตย์'
    }
    day_in_thai = days_translation[day_of_week]
    return f"วันนี้คือ {day_in_thai} ค่ะ"

#------------------------------------------------------------------------------------------------

#การตอบสนองต่อคำถาม
def get_response(intents_list, intents_json):
    if intents_list:
        tag = intents_list[0]['intents']
        list_of_intents = intents_json['intents']
        for i in list_of_intents:
            if i['tag'] == tag:
                if tag == "ask_time":
                    result = get_current_time()
                elif tag == "ask_date":
                    result = get_current_date()
                elif tag == "ask_day":
                    result = get_current_day()
                else:
                    result = random.choice(i['responses'])
                return result
    else:
        return "ขอโทษค่ะ ฉันไม่เข้าใจคำถามของคุณ กรุณาลองใหม่อีกครั้ง"


print("-----------------")
print("Start bot system")
print("-----------------")

while True:
    message = input("")
    ints = predict_class(message)
    res = get_response(ints, intents)
    print(res)

