import pandas as pd
import numpy as np
import streamlit as st

from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsRegressor
from sklearn import tree
from sklearn.model_selection import train_test_split

import pickle
from os.path import exists

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
        "Aplikasi ini dibuat untuk memprediksi harga saham PT GoTo Gojek Tokopedia Tbk (GOTO.JK). Dataset untuk pembuatan aplikasi ini berbentuk time series yang didapat dari laman finance.yahoo.com"
    )
    st.write(
        "Data yang digunakan adalah harga saham PT. GoTo Gojek Tokopedia Tbk (GOTO.JK) dalam kurun waktu tertentu."
    )
    st.write(
        "Repository aplikasi ini dapat diakses pada link berikut: https://github.com/andreandriand/uas-prosainsdata"
    )

    st.write('Kelompok :')
    st.write('1. Andrian Dwi Baitur Rizky (200411100210)')
    st.write('2. Muhammad Aulia Faqihuddin (200411100027)')

    st.subheader("Dataset Saham PT GoTo Gojek Tokopedia Tbk (GOTO.JK)")

    st.write(data)

with tab2:
    st.subheader("Preprocessing Data")
    st.write("Data preprocessing adalah teknik yang digunakan untuk mengubah data mentah dalam format yang berguna dan efisien. Inisiatif ini diperlukan karena data mentah seringkali tidak lengkap dan memiliki format yang tidak konsisten. Preprocessing melibatkan proses validasi dan imputasi data. Validasi bertujuan untuk menilai tingkat kelengkapan dan akurasi data yang tersaring. Sedangkan imputasi bertujuan memperbaiki kesalahan dan memasukkan nilai yang hilang, baik secara manual atau otomatis.")
    st.write("Silahkan pilih metode preprocessing yang ingin digunakan, kemudian tekan tombol Preprocessing untuk memulai proses data preprocessing.")

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

        df_X = pd.DataFrame(X, columns=["t-" + str(i) for i in range(2 - 1, -1, -1)])
        df_y = pd.DataFrame(y, columns=["t+1 (prediction)"])
        df_y.to_csv("data_y.csv", index=False)

        df_data = pd.concat([df_X, df_y], axis=1)
        if prep == "MinMax Scaler":
            scaler = MinMaxScaler()

            X_norm = scaler.fit_transform(df_X)

            scalerFile = 'scaler.sav'
            pickle.dump(scaler, open(scalerFile, 'wb'))

            st.write("MinMax Scaler")

        elif prep == "Reduksi Dimensi":
            scaler = StandardScaler()
            scaler.fit(df_X)
            X_scaled = scaler.transform(df_X)

            pca = PCA(n_components=2)
            pca.fit(X_scaled)
            X_norm = pca.transform(X_scaled)

            pcaFile = 'pca.sav'
            pickle.dump(pca, open(pcaFile, 'wb'))

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

    st.write('Setelah melalui proses preprocessing data, langkah berikutnya adalah pembentukan model (Modelling). Silahkan pilih model yang ingin digunakan, kemudian tekan tombol "Modelling" untuk memulai proses modelling.')

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

            filename_knn = 'model_knn.sav'
            pickle.dump(model_knn, open(filename_knn, 'wb'))

            y_pred = model_knn.predict(X_test)

            mape = mean_absolute_percentage_error(y_test, y_pred)

        elif model == "Naive Bayes":
            nb = GaussianNB()
            nb.fit(X_train, y_train)

            nb_mms = 'model_nb.sav'
            pickle.dump(nb, open(nb_mms, 'wb'))

            y_pred_nb = nb.predict(X_test)

            mape = mean_absolute_percentage_error(y_test, y_pred_nb)

        elif model == "Decision Tree":
            model_tree = tree.DecisionTreeClassifier()
            model_tree.fit(X_train, y_train)
            tree = 'model_tree.sav'
            pickle.dump(model_tree, open(tree, 'wb'))
            y_pred_tree = model_tree.predict(X_test)

            mape = mean_absolute_percentage_error(y_test, y_pred_tree)

        else:
            st.write("Tidak ada preprocessing yang dipilih")

        st.subheader("Akurasi Model")
        st.write("Berikut adalah akurasi model yang anda pilih:")
        st.write("Mean Absolute Percentage Error: ", round(100 * mape, 2), "%")

with tab4:
    st.header("Implementasi Aplikasi")
    st.write(
        'Silahkan isi input dibawah ini dengan benar. Setelah itu tekan tombol "Prediksi" untuk memprediksi'
    )

    if exists('scaler.sav'):
        scaler = pickle.load(open('scaler.sav', 'rb'))
    if exists('pca.sav'):
        pca = pickle.load(open('pca.sav', 'rb'))

    with st.form(key="Form4"):
        input1 = st.number_input("Masukkan Harga Saham (Close) Kemarin", min_value=0)
        input2 = st.number_input("Masukkan Harga Saham (Close) Sekarang", min_value=0)

        int1 = int(input1)
        int2 = int(input2)

        submitted3 = st.form_submit_button(label="Prediksi")

    if submitted3:
        if prep == "MinMax Scaler":
            input_norm = scaler.transform(np.array([[int1, int2]]))
        elif prep == "Reduksi Dimensi":
            input_norm = pca.transform(np.array([[int1, int2]]))
        else:
            st.write("Tidak ada preprocessing yang dipilih")


        if model == "K-Nearest Neighbors":
            model_knn = pickle.load(open('model_knn.sav', 'rb'))
            y_pred = model_knn.predict(input_norm)          

        elif model == "Naive Bayes":
            model_nb = pickle.load(open('model_nb.sav', 'rb'))
            y_pred = model_nb.predict(input_norm)

        elif model == "Decision Tree":
            model_tree = pickle.load(open('model_tree.sav', 'rb'))    
            y_pred = model_tree.predict(input_norm)
        
        else:
            st.write("Tidak ada Model yang sesuai")

        st.write("Prediksi Harga Saham (Close) Besok: ", y_pred[0])
        st.write("Model yang digunakan: ", model)
        st.write("Preprocessing yang digunakan: ", prep)