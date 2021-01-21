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
    51: 500, 52: 500, 53: 500, 54: 500, 55: 500,
    56: 500, 57: 500, 58: 500, 59: 500, 60: 500,
    61: 500
}


def typicalsamling(group, typicalNDict):
    name = group.name
    n = typicalNDict[name]
    return group.sample(n=n)


major = pd.read_csv(r"C:\Users\钟顺民\Desktop\handbags.csv", sep=",", encoding='ISO-8859-1') \
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
c = pd.read_csv(r"C:\Users\钟顺民\Desktop\handbags.csv", sep=',', encoding='ISO-8859-1').dropna().sample(n=200)

prediction = pipeline.predict(c.drop(['id'], axis=1))
t = c['id']
print("Accuracy Score ->", accuracy_score(prediction, t) * 100)
"""
Accuracy Score -> 98.0
"""
# print(accuracy_score(prediction, Test_Y) * 100)
# Test_X.to_csv(r"C:\Users\钟顺民\Desktop\test.csv")
# df.to_csv(r"C:\Users\钟顺民\Desktop\3.csv")
