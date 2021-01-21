import pandas as pd
import numpy as np

np.random.seed(10)

typicalNDict_Shoes = {
    11: 1500, 12: 1500, 13: 1500, 14: 1500, 15: 1500,
    16: 1500, 17: 1500, 18: 1500, 19: 1500
}


def typicalsamling(group, typicalNDict):
    name = group.name
    n = typicalNDict[name]
    return group.sample(n=n)


def get_shoes_data():
    data = pd.read_csv(r"..\..\..\data\klarna\shoes.csv", na_values='na')
    return data


if __name__ == '__main__':
    shoes_data = get_shoes_data()
    shoes_data = shoes_data.groupby('id', as_index=False, group_keys=False) \
        .apply(typicalsamling, typicalNDict_Shoes).reset_index().drop(["index"], axis=1)

    shoes_data.to_csv(r"../../../sample/shoes_sample.csv")


