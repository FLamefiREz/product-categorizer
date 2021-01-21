import pandas as pd
import numpy as np
from nltk.tokenize import word_tokenize
from nltk import pos_tag
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from collections import defaultdict
from nltk.corpus import wordnet as wn

np.random.seed(10)

typicalNDict_Major = {
    31: 1955, 32: 4840, 33: 8357, 34: 148, 35: 2738, 36: 11496, 37: 3414, 38: 1624, 39: 2518, 40: 3693, 41: 1387, 42: 32949, 43: 266, 45: 3262, 46: 181, 47: 573
}


def typicalsamling(group, typicalNDict):
    name = group.name
    n = typicalNDict[name]
    return group.sample(n=n)


def get_data():
    data = pd.read_csv(r"C:\Users\钟顺民\Desktop\2.csv", encoding='ISO-8859-1').groupby('id', as_index=False, group_keys=False) \
    .apply(typicalsamling, typicalNDict_Major)
    return data


def get_sample(data):
    tag_map = defaultdict(lambda: wn.NOUN)
    tag_map['J'] = wn.ADJ
    tag_map['V'] = wn.VERB
    tag_map['R'] = wn.ADV

    result = pd.DataFrame()

    for index, entry in enumerate(data):
        Final_words = []
        word_Lemmatized = WordNetLemmatizer()

        for word, tag in pos_tag(entry):
            if word not in stopwords.words('english') and word.isalpha():
                word_Final = word_Lemmatized.lemmatize(word, tag_map[tag[0]])
                Final_words.append(word_Final)
            result.loc[index, 'word'] = str(Final_words)
    return result


if __name__ == '__main__':
    macys_data = get_data()

    names = macys_data["name"]
    descriptions = macys_data["description"]
    ds = macys_data["id"]

    descriptions.dropna(inplace=True)
    descriptions = [entry.lower() for entry in descriptions]
    descriptions = [word_tokenize(entry) for entry in descriptions]
    descriptions = get_sample(descriptions)['word']

    names.dropna(inplace=True)
    names = [entry.lower() for entry in names]
    names = [word_tokenize(entry) for entry in names]
    names = get_sample(names)['word']

    ids = macys_data["id"]
    samples = pd.DataFrame({"names": names.values, "descriptions": descriptions.values, "ids": ids.values})
    samples.to_csv(r"C:\Users\钟顺民\Desktop\total_data.csv")
