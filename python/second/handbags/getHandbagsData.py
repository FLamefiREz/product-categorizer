import pandas as pd
import numpy as np


np.random.seed(10)

typicalNDict_HandBags = {
    51: 500, 52: 500, 53: 500, 54: 500, 55: 500,
    56: 500, 57: 500, 58: 500, 59: 500, 60: 500,
    61: 500
}


def typicalsamling(group, typicalNDict):
    name = group.name
    n = typicalNDict[name]
    return group.sample(n=n)


def get_shoes_data():
    data = pd.read_csv(r"..\..\..\data\klarna\handbags.csv", na_values='na')
    return data


if __name__ == '__main__':
    handBags_data = get_shoes_data()
    handBags_data = handBags_data.groupby('id', as_index=False, group_keys=False) \
        .apply(typicalsamling, typicalNDict_HandBags).reset_index().drop(["index"], axis=1)

    handBags_data.to_csv(r"..\..\..\sample\handbags_sample.csv")

