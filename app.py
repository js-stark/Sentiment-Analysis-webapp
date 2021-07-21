# -*- coding: utf-8 -*-
"""
Created on Tue Jul 20 19:51:15 2021

@author: jsann
"""

# -*- coding: utf-8 -*-
"""
Created on Mon Jul 19 08:12:15 2021

@author: jsann
"""

from flask import Flask, render_template, flash, request, url_for, redirect, session
import numpy as np
import pandas as pd
import re
import os
import preprocess_js
import tensorflow as tf
from numpy import array
from tensorflow.keras.datasets import imdb
from tensorflow.keras.preprocessing import sequence
from tensorflow.keras.models import load_model


image = os.path.join('static', 'emotion_ms')

app = Flask(__name__)

app.config['UPLOAD_FOLDER'] = image

def init():
    global model
    
    model = load_model('my_model.h5')
    #graph = tf.get_default_graph()

@app.route('/', methods=["GET","POST"])
def home():
    return render_template('Index.html')


@app.route('/sentiment_analysis_prediction',methods = ["POST","GET"])
def sent_anly_prediction():
    if request.method == "POST":
        text = request.form['text']
        sentiment = ''
        max_review_length = 800
        word_to_id = imdb.get_word_index()
        text = preprocess_js.make_base(text)
        text = preprocess_js.remove_accented_chars(text)
        text = preprocess_js.remove_html_tags(text)
        text = preprocess_js.remove_rt(text)
        text = preprocess_js.remove_special_chars(text)
        text = preprocess_js.remove_urls(text)
        text = preprocess_js.remove_urls(text)
                
        words = text.split()
        X_test = [[word_to_id[word] if (word in word_to_id and word_to_id[word]<=20000) else 0 for word in words]]
        X_test = sequence.pad_sequences(X_test,maxlen=max_review_length)
        output = np.array([X_test.flatten()])
        
        
        result=model.predict(output)
        probability = (result[0][0])
        class_predicted =round(probability*100)
        
        if class_predicted <=5:
            sentiment = 'very sad'
            img_filename = os.path.join(app.config['UPLOAD_FOLDER'],'very_sad.png')
            
        elif class_predicted >=5 and class_predicted <=10:
            sentiment = 'sad'
            img_filename = os.path.join(app.config['UPLOAD_FOLDER'],'sad.png')
            
        elif class_predicted >=10 and class_predicted <=20:
            sentiment = 'confused'
            img_filename = os.path.join(app.config['UPLOAD_FOLDER'],'confused.png')
        elif class_predicted >=20 and class_predicted <=30:
            sentiment = 'angry'
            img_filename = os.path.join(app.config['UPLOAD_FOLDER'],'angry.png')
            
        elif class_predicted >=30 and class_predicted <=40:
            sentiment = 'moody'
            img_filename = os.path.join(app.config['UPLOAD_FOLDER'],'moody.png')
            
        elif class_predicted >=40 and class_predicted <=50:
            sentiment = 'pleasant'
            img_filename = os.path.join(app.config['UPLOAD_FOLDER'],'pleasant.png')
            
        elif class_predicted >=50 and class_predicted <=60:
            sentiment = 'studious'
            img_filename = os.path.join(app.config['UPLOAD_FOLDER'],'studious.png')
            
        elif class_predicted >=60 and class_predicted <=75:
            sentiment = 'anxious'
            img_filename = os.path.join(app.config['UPLOAD_FOLDER'],'anxious.png')
        else:
            sentiment = 'very_happy'
            img_filename = os.path.join(app.config['UPLOAD_FOLDER'],'very_happy.png')
    
    return render_template('Index.html', scroll="output",text=text, sentiment=sentiment, probability=probability, image=img_filename)

if __name__ == "__main__":
    init()
    app.run()
        
        
        
        
        
        
        