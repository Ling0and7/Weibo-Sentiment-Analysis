import pandas as pd

from sklearn.model_selection import train_test_split

from tensorflow.keras.preprocessing.text import Tokenizer

from tensorflow.keras.preprocessing.sequence import pad_sequences

from tensorflow.keras.models import Sequential

from tensorflow.keras.layers import Embedding
from tensorflow.keras.layers import LSTM
from tensorflow.keras.layers import Dense

df = pd.read_csv("data/weibo_senti_100k.csv")

texts = df["review"].astype(str)

labels = df["label"]

X_train,X_test,y_train,y_test = train_test_split(
    texts,
    labels,
    test_size=0.2
)

tokenizer = Tokenizer(
    num_words=10000
)

tokenizer.fit_on_texts(X_train)

X_train = tokenizer.texts_to_sequences(X_train)

X_test = tokenizer.texts_to_sequences(X_test)

X_train = pad_sequences(
    X_train,
    maxlen=100
)

X_test = pad_sequences(
    X_test,
    maxlen=100
)

model = Sequential()

model.add(
    Embedding(
        10000,
        128
    )
)

model.add(
    LSTM(128)
)

model.add(
    Dense(
        1,
        activation="sigmoid"
    )
)

model.compile(
    optimizer="adam",
    loss="binary_crossentropy",
    metrics=["accuracy"]
)

model.fit(
    X_train,
    y_train,
    epochs=5,
    batch_size=64
)

model.save(
    "models/lstm_model.keras"
)