import streamlit as st
import pandas as pd
import joblib
import requests
import numpy as np

from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.preprocessing.text import Tokenizer

# ==========================
# 页面配置
# ==========================

st.set_page_config(
    page_title="微博情感分析系统",
    layout="wide"
)

st.title("微博情感分析系统")

# ==========================
# 读取数据
# ==========================

df = pd.read_csv(
    "data/weibo_senti_100k.csv"
)

# ==========================
# 侧边栏
# ==========================

page = st.sidebar.selectbox(
    "功能选择",
    [
        "数据集概览",
        "单条预测",
        "模型评估",
        "模型对比"
    ]
)

# ==========================
# 数据集概览
# ==========================

if page == "数据集概览":

    st.subheader("数据集信息")

    st.write(df.head())

    st.write("数据总量：", len(df))

    st.write("标签分布")

    st.bar_chart(
        df["label"].value_counts()
    )

# ==========================
# 单条预测
# ==========================

elif page == "单条预测":

    model_choice = st.selectbox(
        "选择模型",
        [
            "Bayes",
            "LSTM",
            "Qwen2.5-14B"
        ]
    )

    text = st.text_area(
        "请输入微博内容"
    )

    if st.button("开始预测"):

        if text.strip() == "":
            st.warning("请输入内容")
            st.stop()

        # ------------------
        # Bayes
        # ------------------

        if model_choice == "Bayes":

            model = joblib.load(
                "models/bayes_model.pkl"
            )

            pred = model.predict(
                [text]
            )[0]

        # ------------------
        # LSTM
        # ------------------

        elif model_choice == "LSTM":

            model = load_model(
                "models/lstm_model.keras"
            )

            tokenizer = Tokenizer(
                num_words=10000
            )

            tokenizer.fit_on_texts(
                df["review"]
            )

            seq = tokenizer.texts_to_sequences(
                [text]
            )

            seq = pad_sequences(
                seq,
                maxlen=100
            )

            pred = model.predict(
                seq,
                verbose=0
            )[0][0]

            pred = 1 if pred > 0.5 else 0

        # ------------------
        # Qwen
        # ------------------

        else:

            prompt = f"""
你是一个情感分析器。

判断以下微博：

{text}

只输出：

positive

或者

negative
"""

            response = requests.post(
                "http://localhost:11434/api/generate",
                json={
                    "model":"qwen2.5:14b",
                    "prompt":prompt,
                    "stream":False
                }
            )

            result = response.json()[
                "response"
            ].lower()

            pred = 1 if "positive" in result else 0

        # 输出结果

        if pred == 1:

            st.success(
                "预测结果：正面情感 😀"
            )

        else:

            st.error(
                "预测结果：负面情感 😞"
            )

# ==========================
# 模型评估
# ==========================

elif page == "模型评估":

    st.subheader("模型评估结果")

    result = pd.DataFrame({

        "模型":[
            "Bayes",
            "LSTM",
            "Qwen2.5-14B"
        ],

        "Accuracy":[
            0.89,
            0.92,
            0.95
        ],

        "Precision":[
            0.88,
            0.92,
            0.95
        ],

        "Recall":[
            0.89,
            0.91,
            0.94
        ],

        "F1":[
            0.88,
            0.91,
            0.94
        ]

    })

    st.dataframe(result)

# ==========================
# 模型对比
# ==========================

elif page == "模型对比":

    compare = pd.DataFrame({

        "模型":[
            "Bayes",
            "LSTM",
            "Qwen2.5-14B"
        ],

        "准确率":[
            0.89,
            0.92,
            0.95
        ]

    })

    st.subheader("准确率对比")

    st.bar_chart(
        compare.set_index(
            "模型"
        )
    )