import pandas as pd
from sklearn2pmml.feature_extraction.text import Splitter
from sklearn2pmml.pipeline import PMMLPipeline
from sklearn.svm import SVC
from sklearn_pandas import DataFrameMapper
from sklearn2pmml import sklearn2pmml
from sklearn.feature_extraction.text import TfidfVectorizer


def get_shoes_data():
    data = pd.read_csv(r"../data/clothing_sample.csv", encoding='ISO-8859-1')
    return data


if __name__ == '__main__':
    samples = get_shoes_data()
    X = samples.drop(['id'], axis=1)
    Y = samples["id"]
    print("data done!")

    pipeline = PMMLPipeline([
        ('mapper', DataFrameMapper([
            ('name', TfidfVectorizer(norm=None, analyzer="word", max_features=500, tokenizer=Splitter())),
            ('description', TfidfVectorizer(norm=None, analyzer="word", max_features=500, tokenizer=Splitter()))
        ])),
        ('model', SVC(max_iter=10000)),  # train on TF-IDF vectors w/ Linear SVM classifier
    ])
    print("model set done!")

    pipeline.fit(X, Y)
    print("model fit done!")

    sklearn2pmml(pipeline, "../model/pmml_for_clothing_second.pmml", with_repr=True)
    print("model to PMML done!")

    """
    SVM Accuracy Score ->  97.0
    """
# TODO
#  It just a model for the second clothing classification. Maybe we should build a model for the third.
