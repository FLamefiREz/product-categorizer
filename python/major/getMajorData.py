import pandas as pd
import numpy as np

np.random.seed(10)

typicalNDict_Major = {
    1: 10000, 2: 10000, 3: 10000,
    4: 10000, 5: 10000, 6: 10000
}


def typicalsamling(group, typicalNDict):
    name = group.name
    n = typicalNDict[name]
    return group.sample(n=n)


def get_data():
    data = pd.read_csv(r"../../data/klarna/klarna 2.csv", na_values='na').drop_duplicates()
    return data


def data_group(data, column):
    data = data.groupby(column, as_index=False, group_keys=False) \
        .apply(typicalsamling, typicalNDict_Major)
    return data


if __name__ == '__main__':
    major_data = get_data()
    major_data = major_data.groupby('id', as_index=False, group_keys=False) \
        .apply(typicalsamling, typicalNDict_Major).dropna().reset_index().drop(['index'], axis=1)
    major_data.to_csv("../../sample/major_sample.csv")
