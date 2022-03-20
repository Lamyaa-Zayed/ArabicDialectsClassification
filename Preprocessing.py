# -*- coding: utf-8 -*-
"""
Created on Tue Mar 15 21:26:54 2022

@author: lamya
"""

import pandas as pd 
import re
#from farasa.segmenter import FarasaSegmenter
from arabert.preprocess import ArabertPreprocessor
# from arabert.preprocess_arabert import never_split_tokens, preprocess
from pyarabic.araby import strip_tashkeel,strip_tatweel
#from transformers import AutoTokenizer, AutoModel
from nltk.tokenize import TweetTokenizer

file = pd.read_csv("full_tweets.csv")
#print(file.isna().sum())

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

file['cleaned_tweet'] = file['tweet'].apply(preprocess_text)

# save data after preprocessing
file.to_csv('full_clean_tweets.csv')

print("Done!!")
