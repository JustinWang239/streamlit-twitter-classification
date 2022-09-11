import streamlit as st
import pandas as pd
import snscrape.modules.twitter as sntwitter
from simpletransformers.classification import ClassificationModel
import numpy as np

x = st.slider('Select a value')
st.write(x, 'squared is', x * x)