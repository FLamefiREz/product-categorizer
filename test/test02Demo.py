import numpy as np
from sklearn.datasets import fetch_20newsgroups
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import Pipeline
from sklearn.linear_model import SGDClassifier
import nltk
from nltk.stem.snowball import SnowballStemmer
from sklearn.pipeline import Pipeline
from sklearn.model_selection import GridSearchCV

# Loading the data set - training data.
twenty_train = fetch_20newsgroups(subset='train', shuffle=True)

# 从文本文件中提取特征
count_vect = CountVectorizer()
X_train_counts = count_vect.fit_transform(twenty_train.data)
# print(X_train_counts)
# TF-IDF
tfidf_transformer = TfidfTransformer()
X_train_tfidf = tfidf_transformer.fit_transform(X_train_counts)
print(X_train_tfidf)
# 机器学习
# 在训练数据上训练朴素贝叶斯（NB）分类器。
clf = MultinomialNB().fit(X_train_tfidf, twenty_train.target)

# 建立管道：通过如下构建管道，我们可以编写更少的代码并完成上述所有操作：
# 名称“ vect”，“ tfidf”和“ clf”是任意的，但将在以后使用。
# 我们将继续使用'text_clf'。
text_clf = Pipeline([('vect', CountVectorizer()),
                     ('tfidf', TfidfTransformer()),
                     ('clf', MultinomialNB())])
text_clf = text_clf.fit(twenty_train.data, twenty_train.target)

# NB分类器的性能
twenty_test = fetch_20newsgroups(subset='test', shuffle=True)
predicted = text_clf.predict(twenty_test.data)
np.mean(predicted == twenty_test.target)

# 训练支持向量机-SVM并计算其性能
text_clf_svm = Pipeline([('vect', CountVectorizer()), ('tfidf', TfidfTransformer()),
                         ('clf-svm',
                          SGDClassifier(loss='hinge', penalty='l2', alpha=1e-3, max_iter=5, random_state=42))])


text_clf_svm = text_clf_svm.fit(twenty_train.data, twenty_train.target)
predicted_svm = text_clf_svm.predict(twenty_test.data)
np.mean(predicted_svm == twenty_test.target)

# 网格搜索
# 在这里，我们正在创建要进行性能调整的参数列表。
# 所有参数名称均以分类器名称开头（记住我们给定的任意名称）。
# 例如 vect__ngram_range; 在这里，我们要告诉您使用unigram和bigrams并选择最佳的。
parameters = {'vect__ngram_range': [(1, 1), (1, 2)], 'tfidf__use_idf': (True, False), 'clf__alpha': (1e-2, 1e-3)}

# 接下来，我们通过传递分类器，参数
# 和n_jobs = -1来创建网格搜索的实例，该分类器告诉您使用用户计算机中的多个内核。

gs_clf = GridSearchCV(text_clf, parameters, n_jobs=-1)
gs_clf = gs_clf.fit(twenty_train.data, twenty_train.target)

# 要查看最佳平均得分和参数，请运行以下代码＃上面的输出应为：NB分类器的精度现在已提高到〜90.6％（不再是天真了！😄）
# 和相应的参数为{ 'clf__alpha'：0.01，'tfidf__use_idf'：True，'vect__ngram_range'：（1、2）}。

parameters_svm = {'vect__ngram_range': [(1, 1), (1, 2)], 'tfidf__use_idf': (True, False),
                  'clf-svm__alpha': (1e-2, 1e-3)}

gs_clf_svm = GridSearchCV(text_clf_svm, parameters_svm, n_jobs=-1)
gs_clf_svm = gs_clf_svm.fit(twenty_train.data, twenty_train.target)

# NLTK
# Removing stop words

text_clf = Pipeline([('vect', CountVectorizer(stop_words='english')), ('tfidf', TfidfTransformer()),
                     ('clf', MultinomialNB())])

# Stemming Code


stemmer = SnowballStemmer("english", ignore_stopwords=True)


class StemmedCountVectorizer(CountVectorizer):
    def build_analyzer(self):
        analyzer = super(StemmedCountVectorizer, self).build_analyzer()
        return lambda doc: ([stemmer.stem(w) for w in analyzer(doc)])


stemmed_count_vect = StemmedCountVectorizer(stop_words='english')

text_mnb_stemmed = Pipeline([('vect', stemmed_count_vect), ('tfidf', TfidfTransformer()),
                             ('mnb', MultinomialNB(fit_prior=False))])

text_mnb_stemmed = text_mnb_stemmed.fit(twenty_train.data, twenty_train.target)

predicted_mnb_stemmed = text_mnb_stemmed.predict(twenty_test.data)

print(np.mean(predicted_mnb_stemmed == twenty_test.target))
