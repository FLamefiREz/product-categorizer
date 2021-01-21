import pandas as pd
import numpy as np

np.random.seed(10)

typicalNDict_Clothing = {
    31: 2000, 32: 2000, 33: 2000, 34: 2000, 35: 2000,
    36: 2000, 37: 2000, 38: 2000, 39: 2000, 40: 2000,
    41: 2000, 42: 2000, 43: 2000, 45: 2000, 46: 2000,
    47: 2000
}


def typicalsamling(group, typicalNDict):
    name = group.name
    n = typicalNDict[name]
    return group.sample(n=n)


def get_clothing_data():
    data = pd.read_csv(r"..\..\..\data\klarna\clothing.csv", na_values='na')
    return data


if __name__ == '__main__':
    clothing_data = get_clothing_data()
    clothing_data = clothing_data.groupby('id', as_index=False, group_keys=False) \
        .apply(typicalsamling, typicalNDict_Clothing)\
        .dropna().reset_index().drop(["index"], axis=1)

    clothing_data.to_csv(r"..\..\..\sample\clothing_sample.csv")
