import pandas as pd
import joblib

from sklearn.model_selection import train_test_split

from sklearn.feature_extraction.text import TfidfVectorizer

from sklearn.naive_bayes import MultinomialNB

from sklearn.pipeline import Pipeline

from sklearn.metrics import accuracy_score
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score
from sklearn.metrics import f1_score

df = pd.read_csv("data/weibo_senti_100k.csv")

X = df["review"]
y = df["label"]

X_train,X_test,y_train,y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42
)

model = Pipeline([
    (
        "tfidf",
        TfidfVectorizer(max_features=10000)
    ),
    (
        "nb",
        MultinomialNB()
    )
])

model.fit(X_train,y_train)

pred = model.predict(X_test)

print("Accuracy:",
      accuracy_score(y_test,pred))

print("Precision:",
      precision_score(y_test,pred))

print("Recall:",
      recall_score(y_test,pred))

print("F1:",
      f1_score(y_test,pred))

joblib.dump(
    model,
    "models/bayes_model.pkl"
)

print("保存成功")