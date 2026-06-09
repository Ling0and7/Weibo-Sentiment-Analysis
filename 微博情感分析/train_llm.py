import pandas as pd
import requests
import json

from sklearn.metrics import accuracy_score
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score
from sklearn.metrics import f1_score
from sklearn.model_selection import train_test_split

# ==========================
# 读取数据
# ==========================

df = pd.read_csv(
    "data/weibo_senti_100k.csv"
)

# 抽样
df = df.sample(
    500,
    random_state=42
)

X = df["review"]
y = df["label"]

_, X_test, _, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42
)

# ==========================
# 调用 Ollama
# ==========================

def predict_sentiment(text):

    prompt = f"""
你是一个情感分析器。

请判断下面微博情感：

{text}

只允许输出：

positive

或者

negative
"""

    response = requests.post(
        "http://localhost:11434/api/generate",
        json={
            "model": "qwen2.5:14b",
            "prompt": prompt,
            "stream": False
        }
    )

    result = response.json()["response"]

    result = result.strip().lower()

    if "positive" in result:
        return 1

    return 0

# ==========================
# 开始测试
# ==========================

predictions = []

total = len(X_test)

for idx, text in enumerate(X_test):

    pred = predict_sentiment(str(text))

    predictions.append(pred)

    print(
        f"[{idx+1}/{total}] 完成"
    )

# ==========================
# 评估
# ==========================

acc = accuracy_score(
    y_test,
    predictions
)

precision = precision_score(
    y_test,
    predictions
)

recall = recall_score(
    y_test,
    predictions
)

f1 = f1_score(
    y_test,
    predictions
)

print("\n==========结果==========")

print("Accuracy :", acc)

print("Precision:", precision)

print("Recall   :", recall)

print("F1       :", f1)