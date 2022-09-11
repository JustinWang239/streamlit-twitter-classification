import streamlit as st
import pandas as pd
import snscrape.modules.twitter as sntwitter
from simpletransformers.classification import ClassificationModel
import numpy as np

st.title('Twitter Toxicity Detector')
user = st.text_input('Enter a twitter handle:', '@jordanbpeterson')
x = st.slider('Select the # of latest tweets to review:')

if st.button('Submit'):
    # Created a list to append all tweet attributes(data)
    attributes_container = []

    # Using TwitterSearchScraper to scrape data and append tweets to list
    for i,tweet in enumerate(sntwitter.TwitterSearchScraper('from:' + user).get_items()):
        if i==100:
            break
        attributes_container.append([tweet.content, tweet.url])
        
    # Creating a dataframe from the tweets list above 
    tweets_df = pd.DataFrame(attributes_container, columns=['Tweets', 'Urls'])

    model = ClassificationModel('roberta', 'trained_model', use_cuda=False)
    eval_set = tweets_df['Tweets'].astype(str)
    results = np.array([])

    if (eval_set.size != 0):
        predictions = model.predict(eval_set.tolist())
        results = np.append(results, np.argmax(predictions[1], axis=1))

    tweets_df['Results'] = results.tolist()
    toxic_df = tweets_df[tweets_df['Results'] != 2]

    classifications = {0: 'Hate Speech', 1: 'Offensive'}
    display_df = toxic_df.replace({'Results': classifications})
    display_df.rename(columns={'Results': 'Type'}, inplace=True)

    st.dataframe(toxic_df['Tweets', 'Urls'].reset_index(drop=True))