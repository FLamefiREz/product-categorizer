import pandas as pd
import numpy as np

np.random.seed(10)

typicalNDict_Accessories = {
    91: 500, 92: 500, 93: 500, 94: 500, 95: 500,
    96: 500, 97: 500, 98: 500, 99: 500, 100: 500,
    101: 500, 102: 500, 103: 500, 104: 500, 105: 500,
    106: 500, 107: 500, 109: 500, 110: 500, 111: 500,
    113: 500, 115: 500, 116: 500, 117: 500, 118: 500,
    120: 500
}


def typicalsamling(group, typicalNDict):
    name = group.name
    n = typicalNDict[name]
    return group.sample(n=n)


def get_accessories_data():
    data = pd.read_csv(r"../../../data/klarna/accessories.csv", na_values='na')
    return data


if __name__ == '__main__':
    accessories_data = get_accessories_data()
    accessories_data = accessories_data.groupby('id', as_index=False, group_keys=False) \
        .apply(typicalsamling, typicalNDict_Accessories) \
        .dropna().reset_index().drop(['index'], axis=1)
    accessories_data.to_csv("../../../sample/accessories_sample.csv")