import re
import joblib
import pandas as pd
from sklearn.pipeline import Pipeline
from sklearn.svm import SVC
from sklearn_pandas import DataFrameMapper
from sklearn.feature_extraction.text import TfidfVectorizer


def get_handbags_data():
    data = pd.read_csv(r"../../../sample/handbags_sample.csv")
    return data


def replace_foreign_characters(s):
    return re.sub(r'[^\x00-\x7f]', r'', s)


if __name__ == '__main__':
    samples = get_handbags_data()
    X = samples.drop(['id'], axis=1)
    X['name'] = X['name'].apply(lambda x: replace_foreign_characters(x))
    X['description'] = X['description'].apply(lambda x: replace_foreign_characters(x))
    Y = samples["id"]
    print("data done!")

    pipeline = Pipeline([
        ('mapper', DataFrameMapper([
            ('name', TfidfVectorizer(norm=None, analyzer="word", max_features=500, stop_words="english")),
            ('description', TfidfVectorizer(norm=None, analyzer="word", max_features=1000, stop_words="english"))
        ])),
        ('model', SVC(max_iter=10000)),  # train on TF-IDF vectors w/ Linear SVM classifier
    ])
    print("model set done!")

    pipeline.fit(X, Y)
    print("model fit done!")

    joblib.dump(pipeline, "../../../model/model_for_handbags_second.joblib")
    print("model to JobLib done!")
    """
    SVM Accuracy Score ->  98.0
    """
