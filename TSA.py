import pandas as pd
import numpy as np
import re
import matplotlib.pyplot as plt

data = pd.read_csv("C:/Users/admin/Documents/ML DL course/Projects/twitter sentiment analysis/train.csv")

data.head()

data = data.drop(labels=["id","label"],axis=1)

data.head(20)

def cleanTxt(text):
    text = re.sub(r'@[\w]*','',text) #removing with @ eg: @user
    text = re.sub(r'#','',text)
    return text


data['new']=data['tweet'].apply(cleanTxt)


data.head(20)

data['new'] =data['new'].str.replace("[^a-zA-Z]", " ")
data.head()

import nltk
#nltk.download('punkt')
tf=pd.DataFrame()
from nltk.tokenize import word_tokenize

tf['tokens']=data['new'].apply(lambda x: word_tokenize(x.lower()))
tf.head()

import nltk
nltk.download('wordnet')
from nltk.stem import WordNetLemmatizer
lemmatizer = WordNetLemmatizer()
tf['tokens'].apply(lambda x: [lemmatizer.lemmatize(i) for i in x])
tf.head()
