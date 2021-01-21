import pandas as pd

from sklearn.pipeline import Pipeline
from sklearn_pandas import DataFrameMapper
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from sklearn.svm import SVC, LinearSVC
from sklearn.naive_bayes import MultinomialNB
from nyoka import skl_to_pmml

# merge name and description un to one colume
macys_data = pd.read_csv("macys5k.tsv", delimiter='\t')

X = macys_data.drop(['category_id'], axis=1)
y = macys_data['category_id']
feature_names = ['name', 'description']
target_name = 'category_id'

pipeline = Pipeline([
    ('mapper', DataFrameMapper([
        ('name', TfidfVectorizer(norm=None, analyzer="word", stop_words="english")),
        ('description', TfidfVectorizer(norm=None, analyzer="word", stop_words="english")),
    ])),
    ('model', LinearSVC()),  # train on TF-IDF vectors w/ Linear SVM classifier
])

pipeline.fit(X, y)

skl_to_pmml(pipeline, feature_names, target_name, "apparel_pmml_nyoka.pmml")
