from webbrowser import get
import streamlit as st
import pandas as pd
import snscrape.modules.twitter as sntwitter
from simpletransformers.classification import ClassificationModel
import numpy as np

# @st.cache
def load_model():
	  return ClassificationModel('roberta', 'trained_model', use_cuda=False)

model = load_model()


@st.cache(show_spinner=False)
def get_tweets(user: str, amount: int) -> pd.DataFrame:
    # Created a list to append all tweet attributes(data)
    attributes_container = []

    # Using TwitterSearchScraper to scrape data and append tweets to list
    for i,tweet in enumerate(sntwitter.TwitterSearchScraper('from:' + user).get_items()):
        if i==amount:
            break
        attributes_container.append([tweet.content, tweet.url])
        
    # Creating a dataframe from the tweets list above 
    tweets_df = pd.DataFrame(attributes_container, columns=['Tweets', 'Urls'])

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

    return display_df.reset_index(drop=True)

st.title('Twitter Toxicity Detector')
user = st.text_input('Enter a twitter handle:', '@scrowder')
amount = st.slider('Select the # of latest tweets to review (select 15-25 if server is slow):', min_value=1, max_value=50)

if st.button('Submit'):
    with st.spinner('Loading...'):
        tweets_df = get_tweets(user, amount)
    if (tweets_df.size == 0):
        st.write('No bad tweets found!')
    else:
        st.dataframe(tweets_df)