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
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import classification_report

# 设置随机种子。
# 如果脚本保持一致，则每次使用此命令可重现相同的结果，
# 否则每次运行都会产生不同的结果。种子可以设置为任意数量。
np.random.seed(500)

"""
typicalNDict = {1: 2000,   2: 2000,   3: 2000,   4: 2000,
                5: 2000,   6: 2000,   11: 2000,  12: 100,
                13: 500,   14: 200,   15: 500,   16: 2000,
                17: 500,   18: 1000,  20: 0,     31: 500,
                32: 500,   33: 1000,  34: 0,     35: 500,
                36: 2000,  37: 1000,  38: 100,   39: 100,
                40: 2000,  41: 200,   42: 2000,  43: 100,
                45: 200,   46: 100,   47: 500,   51: 100,
                52: 0,     53: 0,     54: 0,     57: 200,
                59: 0,     60: 100,   61: 200,   72: 2000,
                75: 2000,  76: 2000,  77: 2000,  78: 2000,
                96: 200,   98: 2000,  100: 0,    102: 600,
                104: 2000, 106: 100,  111: 2000, 120: 0,
                121: 0,    122: 100,  123: 200,  124: 100,
                125: 1000, 126: 0,    127: 0,    128: 100,
                312: 100,  333: 200,  336: 200,  361: 500,
                401: 200,  421: 2000, 422: 500,  423: 200,
                452: 200,  1301: 500, 1302: 200, 9000: 0}

typicalNDict = {1: 100,   2: 100,    3: 100,    4: 100,
                5: 100,   6: 100,    11: 100,   12: 100,
                13: 100,  14: 100,   15: 100,   16: 100,
                17: 100,  18: 100,   20: 0,     31: 100,
                32: 100,  33: 100,   34: 0,     35: 100,
                36: 100,  37: 100,   38: 100,   39: 100,
                40: 100,  41: 100,   42: 100,   43: 100,
                45: 100,  46: 100,   47: 100,   51: 100,
                52: 0,    53: 0,     54: 0,     57: 100,
                59: 0,    60: 100,   61: 100,   72: 100,
                75: 100,  76: 100,   77: 100,   78: 100,
                96: 100,  98: 100,   100: 0,    102: 100,
                104: 100, 106: 100,  111: 100,  120: 0,
                121: 0,   122: 100,  123: 100,  124: 100,
                125: 100, 126: 0,    127: 0,    128: 100,
                312: 100, 333: 100,  336: 100,  361: 100,
                401: 100, 421: 100,  422: 100,  423: 100,
                452: 100, 1301: 100, 1302: 100, 9000: 0}


def typicalsamling(group, typicalNDict):
    name = group.name
    n = typicalNDict[name]
    return group.sample(n=n)
"""

# 导入csv数据
Corpus = pd.read_csv(r"C:\Users\钟顺民\Desktop\Text-Classification-master\Text-Classification-master\macys\test.csv")

# 提取数据
names = Corpus['name']
descriptions = Corpus['description']
category_ids = Corpus['category_id']

# 对description字段进行预处理
descriptions.dropna(inplace=True)
descriptions = [entry.lower() for entry in descriptions]
descriptions = [word_tokenize(entry) for entry in descriptions]

# 对names字段进行预处理
names.dropna(inplace=True)
names = [entry.lower() for entry in names]
names = [word_tokenize(entry) for entry in names]

# 将names和description字段组合
for x in range(0, len(names)):
    names[x] = names[x] + descriptions[x]

pd.DataFrame(names).to_csv(r"C:\Users\钟顺民\Desktop\Text-Classification-master\Text-Classification-master\macys\names.csv")
# 将数据中的停止字、非数字字和执行词干/词缀删除
# WordNetLemmatizer要求Pos标签来了解单词是名词，动词还是形容词等。
# 默认情况下，它设置为Noun
tag_map = defaultdict(lambda: wn.NOUN)
tag_map['J'] = wn.ADJ
tag_map['V'] = wn.VERB
tag_map['R'] = wn.ADV

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
        Corpus.loc[index, 'text_final'] = str(Final_words)

result = Corpus.drop(['name', 'description'], axis=1)
result.to_csv(r"C:\Users\钟顺民\Desktop\macys\test02.csv")
"""
tuned_parameters = [{'kernel': ['rbf'], 'gamma': [1e-3, 1e-4],
                     'C': [1, 10, 100, 1000]},
                    {'kernel': ['linear'], 'C': [1, 10, 100, 1000]}]

scores = ['precision', 'recall']

for score in scores:
    print("# Tuning hyper-parameters for %s" % score)
    print()

    clf = GridSearchCV(
        svm.SVC(), tuned_parameters, scoring='%s_macro' % score
    )
    clf.fit(Train_X_Tfidf, Train_Y)

    print("Best parameters set found on development set:")
    print()
    print(clf.best_params_)
    print()
    print("Grid scores on development set:")
    print()
    means = clf.cv_results_['mean_test_score']
    stds = clf.cv_results_['std_test_score']
    for mean, std, params in zip(means, stds, clf.cv_results_['params']):
        print("%0.3f (+/-%0.03f) for %r"
              % (mean, std * 2, params))
    print()

    print("Detailed classification report:")
    print()
    print("The model is trained on the full development set.")
    print("The scores are computed on the full evaluation set.")
    print()
    y_true, y_pred = Test_Y, clf.predict(Test_X_Tfidf)
    print(classification_report(y_true, y_pred))
    print()
"""
# deep learning
"""

"""