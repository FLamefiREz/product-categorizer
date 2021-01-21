import pandas as pd
from sklearn.metrics import accuracy_score
from sklearn2pmml.feature_extraction.text import Splitter
from sklearn2pmml.pipeline import PMMLPipeline
from sklearn.svm import SVC
from sklearn_pandas import DataFrameMapper
from sklearn.feature_extraction.text import TfidfVectorizer


def get_major_data():
    data = pd.read_csv(r"../data/major_sample.csv", encoding='ISO-8859-1')
    return data


if __name__ == '__main__':
    samples = get_major_data()
    X = samples.drop(['id'], axis=1)
    Y = samples["id"]

    pipeline = PMMLPipeline([
        ('mapper', DataFrameMapper([
            ('name', TfidfVectorizer(norm=None, analyzer="word", max_features=1000, tokenizer=Splitter())),
            ('description', TfidfVectorizer(norm=None, analyzer="word", max_features=1000, tokenizer=Splitter()))
        ])),
        ('model', SVC(max_iter=10000)),  # train on TF-IDF vectors w/ Linear SVM classifier
    ])
    print("model set done!")

    pipeline.fit(X, Y)
    print("model fit done!")

    c = pd.read_csv(r"../data/klarna 2.csv", encoding='ISO-8859-1').sample(n=200)

    prediction = pipeline.predict(c.drop(['id'], axis=1))
    t = c['id']
    print("Accuracy Score ->", accuracy_score(prediction, t) * 100)
