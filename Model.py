# -*- coding: utf-8 -*-
"""
Created on Tue Mar 15 22:03:46 2022

@author: lamya
"""
import pandas as pd
from sklearn.model_selection import train_test_split
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from keras.models import Sequential
from keras.layers import Dense, Embedding, LSTM, SpatialDropout1D#, Dropout
from keras.callbacks import EarlyStopping
import pickle

full_clean_tweets = pd.read_csv("full_clean_tweets.csv")
#print(full_clean_tweets.isna().sum())
#full_data_clean_sampled = full_clean_tweet.sample(50000) #take punch of data if want to go with ml models

# The maximum number of words to be used. (most frequent)
MAX_NB_WORDS = 50000
# Max number of words in each complaint.
MAX_SEQUENCE_LENGTH = 150
# This is fixed.
EMBEDDING_DIM = 64
#Tokenizer
tokenizer = Tokenizer(num_words=MAX_NB_WORDS)#, filters='!"#$%&()*+,-./:;<=>?@[\]^_`{|}~', lower=True)
tokenizer.fit_on_texts(full_clean_tweets['cleaned_tweet'].values)
word_index = tokenizer.word_index
X = tokenizer.texts_to_sequences(full_clean_tweets['cleaned_tweet'].values)
X = pad_sequences(X, maxlen=MAX_SEQUENCE_LENGTH)

#OHE for target
Y = pd.get_dummies(full_clean_tweets['dialect']).values

#Split data
X_train, X_test, Y_train, Y_test = train_test_split(X,Y, test_size = 0.20, random_state = 42, shuffle=True, stratify=Y)

#LSTM Model
model = Sequential()
model.add(Embedding(MAX_NB_WORDS, EMBEDDING_DIM, input_length=X.shape[1]))
model.add(SpatialDropout1D(0.2))
model.add(LSTM(100, dropout=0.2, recurrent_dropout=0.2))
model.add(Dense(18, activation='softmax'))
model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
#print(model.summary())

#Train model
epochs = 3
batch_size = 64
history = model.fit(X_train, Y_train, epochs=epochs, batch_size=batch_size,validation_split=0.2,callbacks=[EarlyStopping(monitor='val_loss', patience=3, min_delta=0.0001)])

#Evaluate model
accr = model.evaluate(X_test,Y_test)
print("accuracy:", accr)

#Save model as pickle file
pickle.dump(model, open('lstm_model.pkl','wb'))

print("Done!!!!")
