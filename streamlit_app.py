import pickle
import numpy as np
import pandas as pd
import streamlit as st

title = 'Prediksi Gaji Dengan Algoritma CatBoost 😸'
subtitle = 'Prediksi gaji pekerjaan apa saja dengan Machine Learning'
footer = 'Made With ❤ By Kelompok 1'
data = pd.read_csv("./datasets/clean_data.csv")

company_placeholder = 'Pilih Perusahaan'

COMPANIES = ['Gojek', 'Shopee', 'Tiket.com', 'Tokopedia', 'Traveloka',
             'Bukalapak', 'Other']
CITIES = ['Jakarta', 'Bandung', 'Semarang', 'Yogyakarta', 'Surabaya',
          'Denpasar', 'Other']
COUNTVEC_DIR = './model/count_vectorizer.pkl'
MODEL_DIR = './model/catboost_model.pkl'

with open(COUNTVEC_DIR, 'rb') as file:
    count_vectorizer = pickle.load(file)

with open(MODEL_DIR, 'rb') as file:
    model = pickle.load(file)


def predict(data: pd.DataFrame):
    counts = count_vectorizer.transform(data.role).toarray().tolist()
    X = np.hstack([counts, data.drop('role', axis=1).values])

    y_pred = model.predict(X).tolist()
    return y_pred


def main():
    st.set_page_config(layout="centered", page_icon=":cat:",
                       page_title='Prediksi Gaji Catboost')
    st.title(title)
    st.write(subtitle)

    if st.checkbox('Tampilkan Datasets!'):
        data

    form = st.form("Detail Pekerjaan")

    role = form.text_input('Role Pekerjaan')
    company = form.selectbox('Perusahaan', [company_placeholder] + COMPANIES)
    city = form.selectbox('Daerah Kota', CITIES)
    other_city = form.text_input('Masukkan Nama Kota Jika Memilih other')
    years_of_exp = form.number_input(
        'Tahun Pengalaman', min_value=0, max_value=30
    )

    valid_input = (
        (role != '')
        & (company != company_placeholder)
        & ((city == 'Other') ^ (other_city == ''))
    )

    submit = form.form_submit_button("Prediksi!")
    if submit:
        if not valid_input:
            st.error('Please fill the form properly')
        else:
            data = {
                'role': role.lower(),
                'company': company.lower(),
                'city': city.lower(),
                'years_of_exp': years_of_exp
            }
            data = pd.Series(data).to_frame(name=0).T
            prediction = predict(data)[0]

            st.success('Prediksi Gaji: RP. %.1f Juta' % prediction)

    st.write(footer)


if __name__ == '__main__':
    main()
