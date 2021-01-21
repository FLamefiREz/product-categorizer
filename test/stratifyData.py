import numpy as np
import pandas as pd
from nltk.tokenize import word_tokenize
from nltk import pos_tag
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from collections import defaultdict
from nltk.corpus import wordnet as wn

np.random.seed(500)

typicalNDict_Major = {
    1: 5000, 2: 5000, 3: 5000, 4: 5000,
    5: 5000, 6: 5000, 11: 0, 12: 0,
    13: 0, 14: 0, 15: 0, 16: 00,
    17: 0, 18: 0, 20: 0, 31: 0,
    32: 0, 33: 0, 34: 0, 35: 0,
    36: 0, 37: 0, 38: 0, 39: 0,
    40: 0, 41: 0, 42: 0, 43: 0,
    45: 0, 46: 0, 47: 0, 51: 0,
    52: 0, 53: 0, 54: 0, 57: 0,
    59: 0, 60: 0, 61: 0, 72: 0,
    75: 0, 76: 0, 77: 0, 78: 0,
    96: 0, 98: 0, 100: 0, 102: 0,
    104: 0, 106: 0, 111: 0, 120: 0,
    121: 0, 122: 0, 123: 0, 124: 0,
    125: 0, 126: 0, 127: 0, 128: 0,
    312: 0, 333: 0, 336: 0, 361: 0,
    401: 0, 421: 0, 422: 0, 423: 0,
    452: 0, 1301: 0, 1302: 0, 9000: 0
}


def typicalsamling(group, typicalNDict):
    name = group.name
    n = typicalNDict[name]
    return group.sample(n=n)


Copus = pd.read_csv(r"C:\Users\钟顺民\Desktop\project\product-categorizer\ai\macys\macys.tsv", sep="\t")

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

result.to_csv(r"C:\Users\钟顺民\Desktop\project\product-categorizer\ai\macys\major.csv")
