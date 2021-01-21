import pandas as pd
import numpy as np

np.random.seed(10)

typicalNDict_BeginBeauty = {
    121: 1500, 122: 1500, 123: 1500, 124: 1500, 125: 1500,
    126: 1500, 127: 1500, 128: 1500
}


def typicalsamling(group, typicalNDict):
    name = group.name
    n = typicalNDict[name]
    return group.sample(n=n)


def get_jewellery_data():
    data = pd.read_csv(r"../../../data/klarna/beginBeauty.csv", na_values='na')
    return data


if __name__ == '__main__':
    beginBeauty_data = get_jewellery_data()
    beginBeauty_data = beginBeauty_data.groupby('id', as_index=False, group_keys=False) \
        .apply(typicalsamling, typicalNDict_BeginBeauty)

    beginBeauty_data.to_csv(r"..\..\..\sample\beginBeauty_sample.csv")