import joblib
import pandas as pd
from sklearn.metrics import accuracy_score

model = joblib.load("test.joblib")
c = pd.read_csv(r"../sample/shoes_samples.csv", sep=',', encoding='ISO-8859-1').dropna().sample(n=200)

prediction = model.predict(c.drop(['id'], axis=1))
t = c['id']
print("Accuracy Score ->", accuracy_score(prediction, t) * 100)