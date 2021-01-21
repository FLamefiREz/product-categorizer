import pandas as pd
import numpy as np
from sklearn.metrics import accuracy_score
from sklearn.pipeline import Pipeline
from sklearn.svm import SVC
from sklearn_pandas import DataFrameMapper
from sklearn.feature_extraction.text import TfidfVectorizer
import re

np.random.seed(1000)

typicalNDict_Major = {
    91: 500, 92: 500, 93: 500, 94: 500, 95: 500,
    96: 500, 97: 500, 98: 500, 99: 500, 100: 500,
    101: 500, 102: 500, 103: 500, 104: 500, 105: 500,
    106: 500, 107: 500, 109: 500, 110: 500, 111: 500,
    113: 500, 115: 500, 116: 500, 117: 500, 118: 500,
    120: 500
}


def typicalsamling(group, typicalNDict):
    name = group.name
    n = typicalNDict[name]
    return group.sample(n=n)


def replace_foreign_characters(s):
    return re.sub(r'[^\x00-\x7f]', r'', s)


major = pd.read_csv(r"../sample/accessories_sample.csv", na_values='na') \
        .dropna().groupby('id', as_index=False, group_keys=False) \
        .apply(typicalsamling, typicalNDict_Major)
major.to_csv("test.csv")

# 分配数据
X = major.drop(['id'], axis=1)
X['name'] = X['name'].apply(lambda x: replace_foreign_characters(x))
X['description'] = X['description'].apply(lambda x: replace_foreign_characters(x))
Y = major["id"]
print("data done!")

pipeline = Pipeline([
    ('mapper', DataFrameMapper([
        ('name', TfidfVectorizer(norm=None, analyzer="word", max_features=5)),
        ('description', TfidfVectorizer(norm=None, analyzer="word", max_features=10))
    ])),
    ('model', SVC(max_iter=1)),  # train on TF-IDF vectors w/ Linear SVM classifier
])
print("model set done!")
pipeline.fit(X, Y)
print("model fit done!")
c = pd.read_csv(r"C:\Users\钟顺民\Desktop\accessories.csv", encoding='ISO-8859-1').dropna().sample(n=200)

prediction = pipeline.predict(c.drop(['id'], axis=1))
t = c['id']
print("Accuracy Score ->", accuracy_score(prediction, t) * 100)

"""
Accuracy Score -> 99.0
"""
