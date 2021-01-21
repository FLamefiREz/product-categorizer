import numpy as np
import pandas as pd
from nltk.tokenize import word_tokenize
from nltk import pos_tag
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from collections import defaultdict
from nltk.corpus import wordnet as wn
from sklearn.preprocessing import LabelEncoder
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn import model_selection, naive_bayes, svm
from sklearn.metrics import accuracy_score


np.random.seed(500)

typicalNDict_Major = {

    1: 10000, 2: 10000, 3: 10000, 4: 10000,
    5: 10000, 6: 10000
}


def typicalsamling(group, typicalNDict):
    name = group.name
    n = typicalNDict[name]
    return group.sample(n=n)


Copus = pd.read_csv(r"C:\Users\钟顺民\Desktop\project\product-categorizer\ai\macys\macys02.tsv", sep="\t")

major = Copus.groupby("category_id", as_index=False, group_keys=False) \
    .apply(typicalsamling, typicalNDict_Major)

names = major["name"]
category_ids = major["category_id"]
descriptions = major["description"]

# 对description字段进行预处理
descriptions.dropna(inplace=True)
descriptions = [entry.lower() for entry in descriptions]
descriptions = [word_tokenize(entry) for entry in descriptions]

# 对names字段进行预处理
names.dropna(inplace=True)
names = [entry.lower() for entry in names]
names = [word_tokenize(entry) for entry in names]

for x in range(0, len(names)):
    names[x] = names[x] + descriptions[x]

tag_map = defaultdict(lambda: wn.NOUN)
tag_map['J'] = wn.ADJ
tag_map['V'] = wn.VERB
tag_map['R'] = wn.ADV

result = pd.DataFrame()

for index, entry in enumerate(names):
    # 声明空列表以存储遵循此步骤规则的单词
    Final_words = []
    # 初始化WordNetLemmatizer()
    word_Lemmatized = WordNetLemmatizer()
    # 下面的pos_tag函数将提供“标签”，即单词是否是名词（N）或动词（V）或其他。
    for word, tag in pos_tag(entry):
        # 下面的条件是检查停用词，
        if word not in stopwords.words('english') and word.isalpha():
            word_Final = word_Lemmatized.lemmatize(word, tag_map[tag[0]])
            Final_words.append(word_Final)
        # 每次迭代的最终处理过的单词集将存储在
        result.loc[index, 'text_final'] = str(Final_words)

result.insert(1, 'category_id', category_ids.values)

# print(Corpus['category_id'])
# 分配数据
Train_X, Test_X, Train_Y, Test_Y = model_selection \
    .train_test_split(result['text_final'], result['category_id'], test_size=0.2)

# 标签对目标变量进行编码-这样做是为了将数据集中的字符串类型的分类数据转换为模型可以理解的数值。
Encoder = LabelEncoder()
Train_Y = Encoder.fit_transform(Train_Y)
Test_Y = Encoder.fit_transform(Test_Y)

# 单词向量化
Tfidf_vect = TfidfVectorizer(max_features=5000)
Tfidf_vect.fit(result['text_final'])
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

'''
Naive Bayes Accuracy Score ->  93.56666666666666
LinearSVM Accuracy Score ->  96.25
SVM Accuracy Score ->  96.78333333333333
'''