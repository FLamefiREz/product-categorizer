import pandas as pd
from sklearn2pmml.feature_extraction.text import Splitter
from sklearn2pmml.pipeline import PMMLPipeline
from sklearn.svm import SVC
from sklearn_pandas import DataFrameMapper
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn2pmml import sklearn2pmml


def get_jewellery_data():
    data = pd.read_csv(r"../data/jewellery_sample.csv", encoding='ISO-8859-1')
    return data


if __name__ == '__main__':
    samples = get_jewellery_data()
    X = samples.drop(['id'], axis=1)
    Y = samples["id"]
    print("data done!")

    pipeline = PMMLPipeline([
        ('mapper', DataFrameMapper([
            ('name', TfidfVectorizer(norm=None, analyzer="word", max_features=200, tokenizer=Splitter())),
            ('description', TfidfVectorizer(norm=None, analyzer="word", max_features=600, tokenizer=Splitter()))
        ])),
        ('model', SVC(max_iter=10000)),  # train on TF-IDF vectors w/ Linear SVM classifier
    ])
    print("model set done!")

    pipeline.fit(X, Y)
    print("model fit done!")

    sklearn2pmml(pipeline, "../model/pmml_for_jewellery_second.pmml", with_repr=True)
    print("model to PMML done!")
    """
    SVM Accuracy Score -> 99.5
    """
# TODO
#     71,73,74,79 need more real data.
