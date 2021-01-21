import pandas as pd
from sklearn.pipeline import Pipeline
from sklearn.svm import SVC
from sklearn_pandas import DataFrameMapper
from sklearn.feature_extraction.text import TfidfVectorizer
import joblib


def get_major_data():
    data = pd.read_csv(r"../sample/shoes_samples.csv", encoding='ISO-8859-1')
    return data


if __name__ == '__main__':
    samples = get_major_data()
    X = samples.drop(['id'], axis=1)
    Y = samples["id"]

    feature_names = ['name', 'description']
    target_name = 'category_id'

    pipeline = Pipeline([
        ('mapper', DataFrameMapper([
            ('name', TfidfVectorizer(norm=None, analyzer="word", max_features=200, stop_words="english")),
            ('description', TfidfVectorizer(norm=None, analyzer="word", max_features=600, stop_words="english")),
        ])),
        ('model', SVC(max_iter=1000)),  # train on TF-IDF vectors w/ Linear SVM classifier
    ])

    pipeline.fit(X, Y)
    joblib.dump(pipeline, "test.joblib")
