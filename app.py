# ====================================================================
# app.py yang sudah dimodifikasi untuk membaca dari CSV lokal
# ====================================================================

import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import pickle
import tensorflow as tf
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_percentage_error
from tensorflow.keras.models import load_model 
from tcn import TCN

# -----------------
# Konfigurasi Halaman
# -----------------
st.set_page_config(
    page_title="Perbandingan Model Prediksi Saham",
    layout="wide"
)

st.title("Perbandingan Kinerja Model LSTM dan TCN untuk Prediksi Harga Saham Bank Mandiri (BMRI)")
st.write("Dibuat oleh: Alwan Dzaki Syaeffudin")

# -----------------
# Fungsi-fungsi Bantuan
# -----------------

# Cache untuk memuat data dari CSV lokal
@st.cache_data
def load_data_from_csv(file_path):
    try:
        # 1. Baca CSV, lewati 2 baris pertama yang berisi header tambahan dari yfinance
        # 2. Atur kolom pertama (indeks 0) sebagai index dan langsung parse sebagai tanggal
        data = pd.read_csv(
            file_path, 
            skiprows=2,           # <-- KUNCI 1: Lewati 2 baris header
            index_col=0,          # <-- KUNCI 2: Gunakan kolom tanggal sebagai index
            parse_dates=True      # <-- KUNCI 3: Ubah index menjadi datetime
        )
        
        # 3. Ganti nama kolom agar bersih
        data.columns = ['Open', 'High', 'Low', 'Close', 'Volume']
        
        # 4. Hapus baris yang mungkin memiliki nilai kosong
        data.dropna(inplace=True)
        
        # 5. Pastikan semua kolom fitur adalah tipe numerik
        for col in data.columns:
            data[col] = pd.to_numeric(data[col], errors='coerce')
        
        data.dropna(inplace=True)

        return data

    except FileNotFoundError:
        st.error(f"Error: File data '{file_path}' tidak ditemukan.")
        return None
    except Exception as e:
        st.error(f"Terjadi error saat memproses file CSV: {e}")
        return None
    
# Cache untuk memuat model
@st.cache_resource
def load_prediction_model(model_path):
    try:
        custom_objects = {'TCN': TCN} if 'tcn' in model_path else None
        model = load_model(model_path, custom_objects=custom_objects, compile=False)
        return model
    except Exception as e:
        st.error(f"Gagal memuat model dari {model_path}. Error: {e}")
        return None

# Fungsi untuk membuat dataset sekuensial
def create_dataset(data, time_steps=60):
    x, y = [], []
    for i in range(time_steps, len(data)):
        x.append(data[i-time_steps:i])
        y.append(data[i, 3])
    return np.array(x), np.array(y)

# Fungsi utama untuk prediksi dan evaluasi
def get_model_predictions(model_name, df):
    model_path = f'models/model_{model_name.lower()}.h5'
    scaler_path = f'models/scaler_{model_name.lower()}.pkl'
    
    model = load_prediction_model(model_path)
    if model is None:
        st.error(f"Model {model_name} tidak dapat dimuat.")
        return None
        
    with open(scaler_path, 'rb') as f:
        scaler = pickle.load(f)

    expected_columns = ['Open', 'High', 'Low', 'Close', 'Volume']
    try:
        features = df[expected_columns].copy()
        for col in expected_columns:
            features[col] = pd.to_numeric(features[col], errors='coerce')
        features.dropna(inplace=True)
        if features.empty:
            st.error("Data menjadi kosong setelah dibersihkan.")
            return None
    except KeyError as e:
        st.error(f"Error: Kolom yang dibutuhkan tidak ditemukan di data. {e}")
        return None
    
    scaled_data = scaler.transform(features)
    test_data = scaled_data[int(len(scaled_data) * 0.8):]
    
    x_test, _ = create_dataset(test_data)
    
    predictions_scaled = model.predict(x_test)
    
    test_data_partial = test_data[60:]
    zero_fill_pred = np.zeros((len(predictions_scaled), scaled_data.shape[1]))
    zero_fill_pred[:, 3] = predictions_scaled.flatten()
    predictions_denormalized = scaler.inverse_transform(zero_fill_pred)[:, 3]
    
    y_test_denormalized = scaler.inverse_transform(test_data_partial)[:, 3]

    r2 = r2_score(y_test_denormalized, predictions_denormalized)
    rmse = np.sqrt(mean_squared_error(y_test_denormalized, predictions_denormalized))
    mape = mean_absolute_percentage_error(y_test_denormalized, predictions_denormalized) * 100

    last_60_days = scaled_data[-60:]
    last_60_days_reshaped = np.reshape(last_60_days, (1, 60, 5))
    predicted_next_day_scaled = model.predict(last_60_days_reshaped)
    
    zero_fill_next = np.zeros((1, scaled_data.shape[1]))
    zero_fill_next[:, 3] = predicted_next_day_scaled.flatten()
    predicted_next_day_price = scaler.inverse_transform(zero_fill_next)[:, 3][0]

    return {
        "dates": df.index[-len(y_test_denormalized):],
        "actual": y_test_denormalized,
        "predicted": predictions_denormalized,
        "r2": r2,
        "rmse": rmse,
        "mape": mape,
        "next_day_prediction": predicted_next_day_price
    }

# -----------------
# Sidebar dan Kontrol
# -----------------
st.sidebar.header("Pilih Model")
model_choice = st.sidebar.selectbox(
    "Pilih model untuk ditampilkan:",
    ("Perbandingan", "LSTM", "TCN")
)

# -----------------
# Memuat Data Utama
# -----------------
# --- Memuat data file CSV lokal ---
csv_file_path = 'data_saham/SAHAM-BMRI.JK.csv'

with st.spinner(f'Memuat data saham dari {csv_file_path}...'):
    df_bmri = load_data_from_csv(csv_file_path)

# -----------------
# Logika Tampilan Utama
# -----------------
if df_bmri is not None:
    st.header("Analisis Data Eksplorasi (EDA)")
    st.write("Berikut adalah grafik harga penutupan historis saham BMRI.")
    st.write("Grafik ini menampilkan harga penutupan saham BMRI dari 16 Februari 2015 hingga 16 Februari 2025.")
    fig_hist, ax_hist = plt.subplots(figsize=(12, 6))
    ax_hist.plot(df_bmri.index, df_bmri['Close'], color='blue')
    ax_hist.set_title('Grafik Harga Historis Saham BMRI')
    ax_hist.set_xlabel('Tahun')
    ax_hist.set_ylabel('Harga Penutupan (IDR)')
    ax_hist.grid(True)
    st.pyplot(fig_hist)

    if model_choice == "Perbandingan":
        st.header("Perbandingan Model LSTM dan TCN")
        
        with st.spinner('Memuat hasil model LSTM...'):
            lstm_results = get_model_predictions("LSTM", df_bmri)
        
        with st.spinner('Memuat hasil model TCN...'):
            tcn_results = get_model_predictions("TCN", df_bmri)
    
        if lstm_results and tcn_results:
            col1, col2 = st.columns(2)
            with col1:
                st.subheader("Hasil Model LSTM")
                fig_lstm, ax_lstm = plt.subplots(figsize=(10, 5))
                ax_lstm.plot(lstm_results["dates"], lstm_results["actual"], color='blue', label='Harga Aktual')
                ax_lstm.plot(lstm_results["dates"], lstm_results["predicted"], color='red', label='Harga Prediksi')
                ax_lstm.set_title('LSTM: Aktual vs Prediksi')
                ax_lstm.legend()
                ax_lstm.grid(True)
                st.pyplot(fig_lstm)
                st.metric(label="R-squared (Akurasi)", value=f"{lstm_results['r2']:.2%}")
                st.metric(label="RMSE", value=f"Rp {lstm_results['rmse']:.2f}")
                st.metric(label="MAPE", value=f"{lstm_results['mape']:.2f}%")
                st.success(f"Prediksi Harga Penutupan Saham BMRI di Hari Berikutnya: Rp {lstm_results['next_day_prediction']:,.2f}")
            with col2:
                st.subheader("Hasil Model TCN")
                fig_tcn, ax_tcn = plt.subplots(figsize=(10, 5))
                ax_tcn.plot(tcn_results["dates"], tcn_results["actual"], color='blue', label='Harga Aktual')
                ax_tcn.plot(tcn_results["dates"], tcn_results["predicted"], color='orange', label='Harga Prediksi')
                ax_tcn.set_title('TCN: Aktual vs Prediksi')
                ax_tcn.legend()
                ax_tcn.grid(True)
                st.pyplot(fig_tcn)
                st.metric(label="R-squared (Akurasi)", value=f"{tcn_results['r2']:.2%}")
                st.metric(label="RMSE", value=f"Rp {tcn_results['rmse']:.2f}")
                st.metric(label="MAPE", value=f"{tcn_results['mape']:.2f}%")
                st.success(f"Prediksi Harga Penutupan Saham BMRI di Hari Berikutnya: Rp {tcn_results['next_day_prediction']:,.2f}")
    else:
        st.header(f"Analisis Model {model_choice}")
        with st.spinner(f'Memuat hasil model {model_choice}...'):
            results = get_model_predictions(model_choice, df_bmri)
        if results:
            st.subheader(f"Grafik Prediksi Model {model_choice}")
            fig, ax = plt.subplots(figsize=(12, 6))
            ax.plot(results["dates"], results["actual"], color='blue', label='Harga Aktual')
            ax.plot(results["dates"], results["predicted"], color='red', label='Harga Prediksi')
            ax.set_title(f'{model_choice}: Harga Aktual vs Prediksi')
            ax.set_xlabel('Tanggal')
            ax.set_ylabel('Harga (IDR)')
            ax.legend()
            ax.grid(True)
            st.pyplot(fig)
            st.subheader("Metrik Evaluasi Model")
            col1, col2, col3 = st.columns(3)
            col1.metric(label="R-squared", value=f"{results['r2']:.2%}")
            col2.metric(label="RMSE", value=f"Rp {results['rmse']:.2f}")
            col3.metric(label="MAPE", value=f"{results['mape']:.2f}%")
            st.success(f"Prediksi Harga Penutupan Saham untuk Hari Berikutnya: Rp {results['next_day_prediction']:,.2f}")
else:
    st.warning("Gagal memuat data dari file CSV. Aplikasi tidak dapat berjalan.")