import pandas as pd

df = pd.read_csv(r"C:\Users\钟顺民\Desktop\1.csv", encoding='ISO-8859-1')
df1 = df.dropna().drop_duplicates().to_csv(r"C:\Users\钟顺民\Desktop\2.csv")
