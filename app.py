import pandas as pd
import numpy as np
import streamlit as st
import plotly.express as px

### Modelling
from sklearn import preprocessing
from sklearn.metrics import (
    confusion_matrix,
    accuracy_score,
    recall_score,
    precision_score,
    f1_score,
)
from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsRegressor
from sklearn import tree
from sklearn.ensemble import VotingClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.utils.validation import joblib

from sklearn.metrics import mean_squared_error
from sklearn.metrics import mean_absolute_percentage_error

tab1, tab2, tab3, tab4 = st.tabs(
    ["Dataset", "Preprocessing", "Modelling", "Implementasi"]
)


def load_data(data):
    df = pd.read_csv(data)
    return df


def split_sequence(sequence, n_steps):
    X, y = list(), list()
    for i in range(len(sequence)):
        end_ix = i + n_steps
        if end_ix > len(sequence) - 1:
            break
        seq_x, seq_y = sequence[i:end_ix], sequence[end_ix]
        X.append(seq_x)
        y.append(seq_y)
    return np.array(X), np.array(y)


data = load_data(
    "https://raw.githubusercontent.com/andreandriand/uas-prosainsdata/main/GOTO.JK.csv?token=GHSAT0AAAAAACDUQLKXA5UKPM2M4427JZL2ZEKLRRA"
)

with tab1:
    st.header(
        "Aplikasi Peramalan Harga Saham Perusahaan PT GoTo Gojek Tokopedia Tbk (GOTO.JK)"
    )

    st.write(
        "Aplikasi ini dibuat untuk memprediksi harga saham PT GoTo Gojek Tokopedia Tbk (GOTO.JK) di  masa depan. Dataset untuk pembuatan aplikasi ini berbentuk timeseries yang didapat dari laman finance.yahoo.com"
    )
    st.write(
        "Data yang digunakan adalah Tanggal dan Close, dimana close merupakan harga saham ketika pasar ditutup."
    )
    st.write(
        "Repository aplikasi ini dapat diakses pada link berikut: https://github.com/andreandriand/uas-prosainsdata"
    )
    st.subheader("Dataset Saham PT GoTo Gojek Tokopedia Tbk (GOTO.JK)")

    st.write(data)

with tab2:
    st.subheader("Preprocessing Data")
    st.write("Hilangkan fitur yang tidak diperlukan:")

    with st.form(key="Form2"):
        prep = st.selectbox(
            "Pilih Metode Preprocessing",
            (
                "MinMax Scaler",
                "Reduksi Dimensi",
            ),
        )
        submitted = st.form_submit_button(label="Preprocessing")

    if submitted:
        X, y = split_sequence(data["Close"], 2)

        # column names to X and y data frames
        df_X = pd.DataFrame(X, columns=["t-" + str(i) for i in range(2 - 1, -1, -1)])
        df_y = pd.DataFrame(y, columns=["t+1 (prediction)"])
        df_y.to_csv("data_y.csv", index=False)

        # concat df_X and df_y
        df_data = pd.concat([df_X, df_y], axis=1)
        if prep == "MinMax Scaler":
            scaler = MinMaxScaler()
            X_norm = scaler.fit_transform(df_X)

            st.write("MinMax Scaler")

        elif prep == "Reduksi Dimensi":
            scaler = StandardScaler()
            scaler.fit(df_X)
            X_scaled = scaler.transform(df_X)

            pca = PCA(n_components=2)
            pca.fit(X_scaled)
            X_norm = pca.transform(X_scaled)

            st.write("Reduksi Dimensi")
        else:
            st.write("Tidak ada preprocessing yang dipilih")

        df_norm = pd.concat(
            [pd.DataFrame(X_norm, columns=["X-1_norm", "X-2_norm"])],
            axis=1,
        )

        df_norm.to_csv("data_norm.csv", index=False)

        st.write("Hasil Preprocessing:")
        st.write(df_norm)

with tab3:
    st.subheader("Modelling Data")

    with st.form(key="Form3"):
        model = st.selectbox(
            "Pilih Model",
            (
                "K-Nearest Neighbors",
                "Naive Bayes",
                "Decision Tree",
            ),
        )
        submitted2 = st.form_submit_button(label="Modelling")

    if submitted2:
        df_normalize = pd.read_csv("data_norm.csv")
        df_y = pd.read_csv("data_y.csv")
        y = df_y["t+1 (prediction)"]
        X_norm = df_normalize[["X-1_norm", "X-2_norm"]]
        X_train, X_test, y_train, y_test = train_test_split(
            X_norm, y, test_size=0.2, random_state=0
        )
        if model == "K-Nearest Neighbors":
            model_knn = KNeighborsRegressor(n_neighbors=3)
            model_knn.fit(X_train, y_train)
            y_pred = model_knn.predict(X_test)

            mse = mean_squared_error(y_test, y_pred)
            mape = mean_absolute_percentage_error(y_test, y_pred)

        elif model == "Naive Bayes":
            nb = GaussianNB()
            nb.fit(X_train, y_train)
            y_pred_nb = nb.predict(X_test)

            mse = mean_squared_error(y_test, y_pred_nb)
            mape = mean_absolute_percentage_error(y_test, y_pred_nb)

        elif model == "Decision Tree":
            model_tree = tree.DecisionTreeClassifier()
            model_tree.fit(X_train, y_train)
            y_pred_tree = model_tree.predict(X_test)

            mse = mean_squared_error(y_test, y_pred_tree)
            mape = mean_absolute_percentage_error(y_test, y_pred_tree)

        else:
            st.write("Tidak ada preprocessing yang dipilih")

        st.subheader("Akurasi Model")
        st.write("Berikut adalah akurasi model yang anda pilih:")
        st.write("Mean Squared Error: ", mse)
        st.write("Mean Absolute Percentage Error: ", mape)

with tab4:
    st.header("Implementasi Aplikasi")
    st.write(
        'Silahkan isi input dibawah ini dengan benar. Setelah itu tekan tombol "Predict" untuk memprediksi'
    )
