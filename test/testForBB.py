import pandas as pd
import numpy as np
from sklearn import model_selection
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import LabelEncoder
from sklearn2pmml.feature_extraction.text import Splitter
from sklearn2pmml.pipeline import PMMLPipeline
from sklearn.svm import SVC, LinearSVC
from sklearn_pandas import DataFrameMapper
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn2pmml import sklearn2pmml


np.random.seed(1000)

typicalNDict_Major = {
    121: 1500, 122: 1500, 123: 1500, 124: 1500, 125: 1500,
    126: 1500, 127: 1500, 128: 1500
}


def typicalsamling(group, typicalNDict):
    name = group.name
    n = typicalNDict[name]
    return group.sample(n=n)


major = pd.read_csv(r"C:\Users\钟顺民\Desktop\6.csv", sep=",", encoding='ISO-8859-1') \
    .dropna().groupby('id', as_index=False, group_keys=False) \
    .apply(typicalsamling, typicalNDict_Major)

# 分配数据
X = major.drop(['id'], axis=1)
Y = major["id"]
print("data done!")

pipeline = PMMLPipeline([
    ('mapper', DataFrameMapper([
        ('name', TfidfVectorizer(norm=None, analyzer="word", max_features=500, tokenizer=Splitter())),
        ('description', TfidfVectorizer(norm=None, analyzer="word", max_features=1000, tokenizer=Splitter()))
    ])),
    ('model', SVC(max_iter=10000)),  # train on TF-IDF vectors w/ Linear SVM classifier
])
print("model set done!")
pipeline.fit(X, Y)
print("model fit done!")
c = pd.read_csv(r"C:\Users\钟顺民\Desktop\6.csv", sep=',', encoding='ISO-8859-1').dropna().sample(n=200)

prediction = pipeline.predict(c.drop(['id'], axis=1))
t = c['id']
print("Accuracy Score ->", accuracy_score(prediction, t) * 100)
"""
Accuracy Score -> 98.5
"""

