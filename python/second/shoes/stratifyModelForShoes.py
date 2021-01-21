import pandas as pd
from sklearn2pmml.feature_extraction.text import Splitter
from sklearn2pmml.pipeline import PMMLPipeline
from sklearn.svm import LinearSVC
from sklearn import model_selection
from sklearn_pandas import DataFrameMapper
from sklearn.preprocessing import LabelEncoder
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn2pmml import sklearn2pmml
from sklearn.metrics import accuracy_score


def get_shoes_data():
    data = pd.read_csv(r"../data/shoes_samples.csv", encoding='ISO-8859-1').dropna()
    return data


if __name__ == '__main__':
    samples = get_shoes_data()
    print("data done!")

    X = samples.drop(['id'], axis=1)
    Y = samples["id"]
    feature_names = ['name', 'description']
    target_name = 'category_id'
    # Train_X, Test_X, Train_Y, Test_Y = model_selection \
    #    .train_test_split(X, Y, test_size=0.2)
    pipeline = PMMLPipeline([
        ('mapper', DataFrameMapper([
            ('name', TfidfVectorizer(norm=None, analyzer="word", max_features=200, tokenizer=Splitter())),
            ('description', TfidfVectorizer(norm=None, analyzer="word", max_features=600, tokenizer=Splitter()))
        ])),
        ('model', LinearSVC(max_iter=100000)),  # train on TF-IDF vectors w/ Linear SVM classifier
    ])

    print("model set done!")

    pipeline.fit(X, Y)
    print("model fit done!")

    sklearn2pmml(pipeline, "../model/pmml_for_shoes_second.pmml", with_repr=True)
    print("model to PMML done!")
    """
    LinearSVM Accuracy Score ->  99.5

    """
# TODO I just train this model with 1301 and 1302 instead of 13. The best model I thik is that, when the prediction
#  equals 13, we send this prediction to another model to calculate that the result is 1301 or 1302.
