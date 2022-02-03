# we have downloaded dataset from keggle
# importing dependencies
import numpy as np
import pandas as pd
import re #regular expression useful for searching a text in a document
from nltk.corpus import stopwords #natural language tool kit stopwords means those words which doesnot add value to the article

from nltk.stem.porter import PorterStemmer #this function is used to stem our words
from sklearn.feature_extraction.text import TfidfVectorizer #used to convert text to feature vector
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score

# print(stopwords.words('English')) These are the stopwords in English language

#Data pre Processing
News_DataSet = pd.read_csv('train.csv')
News_DataSet.head()