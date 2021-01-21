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

np.random.seed(100)
typicalNDict_Major = {
    31: 2000, 32: 2000, 33: 2000, 34: 2000, 35: 2000,
    36: 2000, 37: 2000, 38: 2000, 39: 2000, 40: 2000,
    41: 2000, 42: 2000, 43: 2000, 45: 2000, 46: 2000,
    47: 2000
}


def typicalsamling(group, typicalNDict):
    name = group.name
    n = typicalNDict[name]
    return group.sample(n=n)


major = pd.read_csv(r"C:\Users\钟顺民\Desktop\22.csv", sep=",", encoding='ISO-8859-1') \
    .dropna().groupby('id', as_index=False, group_keys=False) \
    .apply(typicalsamling, typicalNDict_Major)

# 分配数据
X = major.drop(['id'], axis=1)
Y = major["id"]

pipeline = PMMLPipeline([
    ('mapper', DataFrameMapper([
        ('name', TfidfVectorizer(norm=None, analyzer="word", max_features=500, tokenizer=Splitter())),
        ('description', TfidfVectorizer(norm=None, analyzer="word", max_features=1000, tokenizer=Splitter()))
    ])),
    ('model', SVC(max_iter=10000)),  # train on TF-IDF vectors w/ Linear SVM classifier
])

pipeline.fit(X, Y)

c = pd.read_csv(r"C:\Users\钟顺民\Desktop\22.csv", sep=',', encoding='ISO-8859-1').dropna().sample(n=200)

prediction = pipeline.predict(c.drop(['id'], axis=1))
t = c['id']
print(accuracy_score(prediction, t) * 100)
# print(accuracy_score(prediction, Test_Y) * 100)
print(c.drop(['id'], axis=1))
print(t.values)
print(prediction)
# Test_X.to_csv(r"C:\Users\钟顺民\Desktop\test.csv")
# df.to_csv(r"C:\Users\钟顺民\Desktop\3.csv")
