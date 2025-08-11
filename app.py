import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import pickle
import tensorflow as tf
from datetime import timedelta
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
@st.cache_data
def load_data_from_csv(file_path):
    try:
        data = pd.read_csv(
            file_path, 
            skiprows=2,
            index_col=0,
            parse_dates=True
        )
        data.columns = ['Open', 'High', 'Low', 'Close', 'Volume']
        data.dropna(inplace=True)
        for col in data.columns:
            data[col] = pd.to_numeric(data[col], errors='coerce')
        data.dropna(inplace=True)
        return data
    except Exception as e:
        st.error(f"Terjadi error saat memproses file CSV: {e}")
        return None

@st.cache_resource
def load_prediction_model(model_path):
    try:
        custom_objects = {'TCN': TCN} if 'tcn' in model_path else None
        model = load_model(model_path, custom_objects=custom_objects, compile=False)
        return model
    except Exception as e:
        st.error(f"Gagal memuat model dari {model_path}. Error: {e}")
        return None

def create_dataset(data, time_steps=60):
    x, y = [], []
    for i in range(time_steps, len(data)):
        x.append(data[i-time_steps:i])
        y.append(data[i, 3])
    return np.array(x), np.array(y)

# --- FUNGSI MENAMPILKAN PREDIKSI 7 HARI ---
def get_model_predictions(model_name, df):
    model_path = f'models/model_{model_name.lower()}.h5'
    scaler_path = f'models/scaler_{model_name.lower()}.pkl'
    
    model = load_prediction_model(model_path)
    if model is None: return None
        
    with open(scaler_path, 'rb') as f:
        scaler = pickle.load(f)

    expected_columns = ['Open', 'High', 'Low', 'Close', 'Volume']
    features = df[expected_columns].copy()
    for col in expected_columns:
        features[col] = pd.to_numeric(features[col], errors='coerce')
    features.dropna(inplace=True)
    if features.empty: return None
    
    scaled_data = scaler.transform(features)
    test_data = scaled_data[int(len(scaled_data) * 0.8):]
    
    x_test, _ = create_dataset(test_data)
    predictions_scaled = model.predict(x_test)
    
    test_data_partial = test_data[60:]
    zero_fill_pred = np.zeros((len(predictions_scaled), 5))
    zero_fill_pred[:, 3] = predictions_scaled.flatten()
    predictions_denormalized = scaler.inverse_transform(zero_fill_pred)[:, 3]
    y_test_denormalized = scaler.inverse_transform(test_data_partial)[:, 3]

    r2 = r2_score(y_test_denormalized, predictions_denormalized)
    rmse = np.sqrt(mean_squared_error(y_test_denormalized, predictions_denormalized))
    mape = mean_absolute_percentage_error(y_test_denormalized, predictions_denormalized) * 100

    # Logika untuk prediksi 7 hari
    future_predictions_list = []
    last_60_days = scaled_data[-60:]
    current_batch = last_60_days.reshape((1, 60, 5))
    for _ in range(7):
        next_pred_scaled = model.predict(current_batch)[0]
        future_predictions_list.append(next_pred_scaled)
        new_row = current_batch[0, -1, :].copy()
        new_row[3] = next_pred_scaled
        current_batch = np.append(current_batch[:, 1:, :], [[new_row]], axis=1)
    
    future_fill = np.zeros((len(future_predictions_list), 5))
    future_fill[:, 3] = np.array(future_predictions_list).flatten()
    future_predictions_denormalized = scaler.inverse_transform(future_fill)[:, 3]
    
    return {
        "dates": df.index[-len(y_test_denormalized):],
        "actual": y_test_denormalized,
        "predicted": predictions_denormalized,
        "r2": r2,
        "rmse": rmse,
        "mape": mape,
        "future_predictions": future_predictions_denormalized
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
csv_file_path = 'data_saham/SAHAM-BMRI.JK.csv'
with st.spinner(f'Memuat data saham dari {csv_file_path}...'):
    df_bmri = load_data_from_csv(csv_file_path)

# -----------------
# Logika Tampilan Utama
# -----------------
if df_bmri is not None:
    st.header("Analisis Data Eksplorasi (EDA)")
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

                st.metric(label="R-squared", value=f"{lstm_results['r2']:.2%}")
                st.metric(label="RMSE", value=f"{lstm_results['rmse']:.2f}")
                st.metric(label="MAPE", value=f"{lstm_results['mape']:.2f}%")
                
                # --- MENAMPILKAN PREDIKSI 7 HARI ---
                st.write("**Perbandingan Harga Aktual dan Harga Prediksi 7 Hari Terakhir**")
                # Membuat DataFrame untuk 7 hari terakhir
                comparison_test_df_lstm = pd.DataFrame({
                    'Harga Aktual (IDR)': lstm_results["actual"][:7],
                    'Prediksi LSTM (IDR)': lstm_results["predicted"][:7]
                }, index=lstm_results["dates"][:7])
                comparison_test_df_lstm.index = comparison_test_df_lstm.index.strftime('%Y-%m-%d')
                st.dataframe(comparison_test_df_lstm.style.format("{:,.2f}"), use_container_width=True)

            with col2:
                st.subheader("Hasil Model TCN")
                fig_tcn, ax_tcn = plt.subplots(figsize=(10, 5))
                ax_tcn.plot(tcn_results["dates"], tcn_results["actual"], color='blue', label='Harga Aktual')
                ax_tcn.plot(tcn_results["dates"], tcn_results["predicted"], color='orange', label='Harga Prediksi')
                ax_tcn.set_title('TCN: Aktual vs Prediksi')
                ax_tcn.legend()
                ax_tcn.grid(True)
                st.pyplot(fig_tcn)
                
                st.metric(label="R-squared", value=f"{tcn_results['r2']:.2%}")
                st.metric(label="RMSE", value=f"{tcn_results['rmse']:.2f}")
                st.metric(label="MAPE", value=f"{tcn_results['mape']:.2f}%")

                # --- MENAMPILKAN PREDIKSI 7 HARI ---
                st.write("**Perbandingan Harga Aktual dan Harga Prediksi 7 Hari Terakhir**")
                # Membuat DataFrame untuk 7 hari terakhir
                comparison_test_df_tcn = pd.DataFrame({
                    'Harga Aktual (IDR)': tcn_results["actual"][:7],
                    'Prediksi TCN (IDR)': tcn_results["predicted"][:7]
                }, index=tcn_results["dates"][:7])
                comparison_test_df_tcn.index = comparison_test_df_tcn.index.strftime('%Y-%m-%d')
                st.dataframe(comparison_test_df_tcn.style.format("{:,.2f}"), use_container_width=True)

    else: # Tampilan untuk model individual
        st.header(f"Analisis Model {model_choice}")
        with st.spinner(f'Memuat hasil model {model_choice}...'):
            results = get_model_predictions(model_choice, df_bmri)
        
        if results:
            st.subheader(f"Grafik Prediksi Model {model_choice}")
            fig, ax = plt.subplots(figsize=(12, 6))
            ax.plot(results["dates"], results["actual"], color='blue', label='Harga Aktual')
            ax.plot(results["dates"], results["predicted"], color='red' if model_choice == 'LSTM' else 'orange', label='Harga Prediksi')
            ax.set_title(f'{model_choice}: Harga Aktual vs Prediksi')
            ax.set_xlabel('Tanggal')
            ax.set_ylabel('Harga (IDR)')
            ax.legend()
            ax.grid(True)
            st.pyplot(fig)
            
            st.subheader("Metrik Evaluasi Model")
            col1, col2, col3 = st.columns(3)
            col1.metric(label="R-squared", value=f"{results['r2']:.2%}")
            col2.metric(label="RMSE", value=f"{results['rmse']:.2f}")
            col3.metric(label="MAPE", value=f"{results['mape']:.2f}%")
            
            # --- MENAMPILKAN PREDIKSI 7 HARI ---
            st.subheader("Perbandingan Harga Aktual dan Harga Prediksi 7 Hari Terakhir")
            # Membuat DataFrame untuk 7 hari terakhir
            test_df = pd.DataFrame({
                'Harga Aktual (IDR)': results["actual"][:7],
                'Harga Prediksi (IDR)': results["predicted"][:7]
            }, index=results["dates"][:7])
            test_df.index = test_df.index.strftime('%Y-%m-%d')
            st.dataframe(test_df.style.format("{:,.2f}"), use_container_width=True)

else:
    st.warning("Gagal memuat data dari file CSV. Aplikasi tidak dapat berjalan.")