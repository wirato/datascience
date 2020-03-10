from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import make_pipeline
from sklearn.datasets import fetch_20newsgroups
import numpy as np

categories = ['comp.os.ms-windows.misc', 'sci.space', 'comp.graphics']

train = fetch_20newsgroups(subset='train', categories=categories)

model = make_pipeline(TfidfVectorizer(), MultinomialNB())

model.fit(train.data, train.target)

from joblib import dump
dump(model, './myapp/static/chatgroup.model')
dump(train, './myapp/static/train.model')

print(round((model.score(train.data, train.target)*100),2),'%')
