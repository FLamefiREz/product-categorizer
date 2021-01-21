import numpy as np
import pandas as pd
from nltk.tokenize import word_tokenize
from nltk import pos_tag
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from sklearn.preprocessing import LabelEncoder
from collections import defaultdict
from nltk.corpus import wordnet as wn
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn import model_selection, naive_bayes, svm
from sklearn.metrics import accuracy_score

np.random.seed(500)

# 导入csv数据
Corpus = pd.read_csv(r"C:\Users\钟顺民\Desktop\Text-Classification-master\Text-Classification-masteru\macys\test02.csv")

print(Corpus['category_id'])
# 分配数据
Train_X, Test_X, Train_Y, Test_Y = model_selection \
    .train_test_split(Corpus['text_final'], Corpus['category_id'], test_size=0.2)

# 标签对目标变量进行编码-这样做是为了将数据集中的字符串类型的分类数据转换为模型可以理解的数值。
Encoder = LabelEncoder()
Train_Y = Encoder.fit_transform(Train_Y)
Test_Y = Encoder.fit_transform(Test_Y)

# 单词向量化
Tfidf_vect = TfidfVectorizer(max_features=5000)
Tfidf_vect.fit(Corpus['text_final'])
Train_X_Tfidf = Tfidf_vect.transform(Train_X)
Test_X_Tfidf = Tfidf_vect.transform(Test_X)

# ML算法预测
Native = naive_bayes.MultinomialNB()
Native.fit(Train_X_Tfidf, Train_Y)
predictions_NB = Native.predict(Test_X_Tfidf)

print("Naive Bayes Accuracy Score -> ", accuracy_score(predictions_NB, Test_Y) * 100)


# SVM算法
LinearSVM = svm.LinearSVC()
LinearSVM.fit(Train_X_Tfidf, Train_Y)
predictions_LinearSVM = LinearSVM.predict(Test_X_Tfidf)

print("LinearSVM Accuracy Score -> ", accuracy_score(predictions_LinearSVM, Test_Y) * 100)

SVM = svm.SVC()
SVM.fit(Train_X_Tfidf, Train_Y)
predictions_SVM = SVM.predict(Test_X_Tfidf)

print("SVM Accuracy Score -> ", accuracy_score(predictions_SVM, Test_Y) * 100)

