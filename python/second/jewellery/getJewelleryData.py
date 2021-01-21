import pandas as pd
import numpy as np

np.random.seed(10)

typicalNDict_Jewellery = {
    71: 5000, 72: 5000, 73: 5000, 74: 5000, 75: 5000,
    76: 5000, 77: 5000, 78: 5000, 79: 5000
}


def typicalsamling(group, typicalNDict):
    name = group.name
    n = typicalNDict[name]
    return group.sample(n=n)


def get_jewellery_data():
    data = pd.read_csv(r"..\..\..\data\klarna\jewellery.csv", na_values='na')
    return data


if __name__ == '__main__':
    jewellery_data = get_jewellery_data()
    jewellery_data = jewellery_data.groupby('id', as_index=False, group_keys=False) \
        .apply(typicalsamling, typicalNDict_Jewellery).dropna().reset_index().drop(["index"], axis=1)

    jewellery_data.to_csv(r"..\..\..\sample\jewellery_sample.csv")
