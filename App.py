# -*- coding: utf-8 -*-
"""
Created on Tue Mar 15 23:18:35 2022

@author: lamya
"""


from flask import Flask, render_template, request
import pickle

import re
#from farasa.segmenter import FarasaSegmenter
from arabert.preprocess import ArabertPreprocessor
from pyarabic.araby import strip_tashkeel,strip_tatweel
from nltk.tokenize import TweetTokenizer
model_name = "aubmindlab/bert-base-arabertv2"
arabert_prep = ArabertPreprocessor(model_name=model_name)
def preprocess_text(tweet):
    outputs=[]
    tokenizer = TweetTokenizer()
    tweet = str(tweet)
    # remove old style retweet text "RT"
    tweet = re.sub(r'^RT[\s]+', '', tweet)
    # remove hyperlinks
    tweet = re.sub(r'https?:\/\/.*[\r\n]*', '', tweet)
    # remove hashtags
    # only removing the hash # sign from the word
    tweet = re.sub(r'#', '', tweet)
    #removing mentions
    tweet = re.sub(r':', '', tweet)
    tweet = re.sub(r'@[\w]+','',tweet)
    #replace punctuations with space
    tweet = re.sub(r"[,.;@#?!&$_]+\ *", " ", tweet)
    #find arabic letters only
    tweet = ' '.join(re.findall(r'[\u0600-\u06FF]+',tweet))
    #remove tashkeel
    tweet = strip_tashkeel(tweet)
    #remove tatweel
    tweet = strip_tatweel(tweet)
    #remove numbers from string
    pattern = r'[0-9]'
    # Match all digits in the string and replace them with an empty string
    tweet = re.sub(pattern, '', tweet)
    #tokenize tweets
    tweet_tokens = tokenizer.tokenize(tweet)
    tweet_clean=[]
    for word in tweet_tokens: # Go through every word in your tokens list
        #if word not in string.punctuation:  # remove punctuation
        word_reg = re.compile(r'\w')
        if word_reg.search(word):
            tweet_clean.append(word)   
    outputs.append((' '.join(tweet_clean)))
    text_preprocessed=arabert_prep.preprocess(outputs)
    return text_preprocessed

#Flask App
app = Flask(__name__)

#Load LSTM Model
lstm_model = pickle.load(open('ls_model.pkl', 'rb'))

#Home
@app.route('/')#, methods=['GET'])
def home():
    return 'Hello World!'
    #return render_template('index.html',variable=None, variables=None)

#Model Prediction
@app.route('/predict',methods=['POST'])
def predict():
    tweet = request.form['text']
    result = list()
    result.append(dict())
    result[0]['tweet'] = tweet
    
    dialect_tweet = preprocess_text(tweet)
    
    dialect_prediction = lstm_model.predict(dialect_tweet)
    
    dialect_proba = lstm_model.predict_proba(dialect_tweet)[0]
    
    #18 Classes: ['EG', 'PL', 'KW', 'LY', 'QA', 'JO', 'LB', 'SA',  
    # 'AE', 'BH', 'OM', 'SY', 'DZ', 'IQ', 'SD', 'MA', 'YE', 'TN']
    
    result[0]['dialect'] = {}
    if dialect_prediction == "EG":
        result[0]['dialect']["EG"] = round(dialect_proba[0]*100, 2)
    if dialect_prediction == "PL":
        result[0]['dialect']["PL"] = round(dialect_proba[1]*100, 2)
    if dialect_prediction == "KW":
        result[0]['dialect']["KW"] = round(dialect_proba[2]*100, 2)
    if dialect_prediction == "LY":
        result[0]['dialect']["LY"] = round(dialect_proba[3]*100, 2)
    if dialect_prediction == "QA":
        result[0]['dialect']["QA"] = round(dialect_proba[4]*100, 2)
    if dialect_prediction == "JO":
        result[0]['dialect']["JO"] = round(dialect_proba[5]*100, 2)
    if dialect_prediction == "LB":
        result[0]['dialect']["LB"] = round(dialect_proba[6]*100, 2)
    if dialect_prediction == "SA":
        result[0]['dialect']["SA"] = round(dialect_proba[7]*100, 2)
    if dialect_prediction == "AE":
        result[0]['dialect']["AE"] = round(dialect_proba[8]*100, 2)
    if dialect_prediction == "BH":
        result[0]['dialect']["BH"] = round(dialect_proba[9]*100, 2)
    if dialect_prediction == "OM":
        result[0]['dialect']["OM"] = round(dialect_proba[10]*100, 2)
    if dialect_prediction == "SY":
        result[0]['dialect']["SY"] = round(dialect_proba[11]*100, 2)
    if dialect_prediction == "DZ":
        result[0]['dialect']["DZ"] = round(dialect_proba[12]*100, 2)
    if dialect_prediction == "IQ":
        result[0]['dialect']["IQ"] = round(dialect_proba[13]*100, 2)
    if dialect_prediction == "SD":
        result[0]['dialect']["SD"] = round(dialect_proba[14]*100, 2)
    if dialect_prediction == "MA":
        result[0]['dialect']["MA"] = round(dialect_proba[15]*100, 2)
    if dialect_prediction == "YE":
        result[0]['dialect']["YE"] = round(dialect_proba[16]*100, 2)
    if dialect_prediction == "TN":
        result[0]['dialect']["TN"] = round(dialect_proba[17]*100, 2)

    return render_template('index.html', variable=result, variables=None)

# App Main
if __name__ == "__main__":
    app.run(debug=False)

#Check    
print("Final Done!")