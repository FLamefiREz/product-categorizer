import pandas as pd
from numpy import shape

df1 = pd.read_csv(r"C:\Users\钟顺民\Desktop\klarna\klarna2.csv", sep="\t", names=["name", "description", "id"])
df2 = pd.read_csv(r"C:\Users\钟顺民\Desktop\klarna\klarna4.csv", sep="\t", names=["name", "description", "id"])
df3 = pd.read_csv(r"C:\Users\钟顺民\Desktop\klarna\klarna5.csv", sep="\t", names=["name", "description", "id"])
df4 = pd.read_csv(r"C:\Users\钟顺民\Desktop\klarna\klarna22.csv", sep="\t", names=["name", "description", "id"])
df5 = pd.read_csv(r"C:\Users\钟顺民\Desktop\klarna\klarna23.csv", sep="\t", names=["name", "description", "id"])
df6 = pd.read_csv(r"C:\Users\钟顺民\Desktop\klarna\klarna27.csv", sep="\t", names=["name", "description", "id"])
df7 = pd.read_csv(r"C:\Users\钟顺民\Desktop\klarna\klarna28.csv", sep="\t", names=["name", "description", "id"])
df8 = pd.read_csv(r"C:\Users\钟顺民\Desktop\klarna\klarna28.csv", sep="\t", names=["name", "description", "id"])
df9 = pd.read_csv(r"C:\Users\钟顺民\Desktop\klarna\klarna29.csv", sep="\t", names=["name", "description", "id"])
df10 = pd.read_csv(r"C:\Users\钟顺民\Desktop\klarna\klarna30.csv", sep="\t", names=["name", "description", "id"])
df11 = pd.read_csv(r"C:\Users\钟顺民\Desktop\klarna\klarna31.csv", sep="\t", names=["name", "description", "id"])
macys = pd.read_csv("../data/macys/macys.tsv", sep="\t",names=["name", "description", "id"])
df = pd.concat([df1, df2, df3, df4, df5, df6, df7, df8, df9, df10, df11,macys]).drop_duplicates().reset_index(drop=True)

c = pd.read_csv(r"../data/klarna/klarna 1.csv", encoding='ISO-8859-1')
print(shape(c))