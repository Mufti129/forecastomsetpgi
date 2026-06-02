import streamlit as st
import pandas as pd
import numpy as np
import joblib
from scipy.spatial.distance import cdist
from pyproj import Transformer
import os

st.set_page_config(page_title="PGI 3-Model Forecasting", layout="wide")

st.title("Aplikasi Simulasi Perbandingan 3 Model Forecasting Omzet PGI")
st.write("Sistem ini membandingkan hasil model OLS, Random Forest, dan Koreksi Spasial GWR secara instan untuk membantu analisis ekspansi cabang.")

# ========================================================
# 1. LOAD OBJEK MODEL DENGAN CACHING AMAN
# ========================================================
@st.cache_resource
def load_all_models():
    # Menggunakan jalur folder absolut aman untuk mendeteksi file .pkl di Streamlit Cloud
    dir_sekarang = os.path.dirname(os.path.abspath(__file__))
    
    path_rf = os.path.join(dir_sekarang, 'rf_model.pkl')
    path_scaler_rf = os.path.join(dir_sekarang, 'scaler_rf.pkl')
    path_ols = os.path.join(dir_sekarang, 'ols_model.pkl')
    path_scaler_ols = os.path.join(dir_sekarang, 'scaler_ols.pkl')
    path_gwr = os.path.join(dir_sekarang, 'gwr_stats.pkl')
    
    model_rf = joblib.load(path_rf)
    scaler_rf = joblib.load(path_scaler_rf)
    model_ols = joblib.load(path_ols)
    scaler_ols = joblib.load(path_scaler_ols)
    gwr_metadata = joblib.load(path_gwr)
    
    return model_rf, scaler_rf, model_ols, scaler_ols, gwr_metadata

try:
    rf_model, scaler_rf, ols_model, scaler_ols, gwr_meta = load_all_models()
    df_ref = gwr_meta['df_spatial_reference']
except Exception as e:
    st.error(f"Gagal memuat komponen model (.pkl). Pastikan semua file pkl sudah di-upload di root repository GitHub Anda. Detail Error: {e}")
    st.stop()

# ========================================================
# 2. FORM INPUT UTAMA PENGGUNA (UI LAYOUT)
# ========================================================
st.markdown("### Masukkan Parameter Karakteristik Cabang Baru")
with st.form("simulation_form"):
    col1, col2 = st.columns(2)
    with col1:
        new_lat = st.number_input("Latitude (Garis Lintang)", value=-6.925914, format="%.6f")
        new_lon = st.number_input("Longitude (Garis Bujur)", value=107.588618, format="%.6f")
        new_umk = st.number_input("UMK Wilayah Cabang (Rp)", value=4482914)
        new_penduduk = st.number_input("Jumlah Penduduk di Wilayah", value=94158)
        new_kemiskinan = st.slider("Proporsi Kemiskinan Wilayah", 0.00, 1.00, 0.04)
        new_lebar_ruko = st.number_input("Lebar Ruko Cabang (cm)", value=450)
        new_jumlah_bangunan = st.number_input("Jumlah Bangunan di Sekitar", value=18840)
    
    with col2:
        new_jumlah_kompetitor = st.number_input("Jumlah Kompetitor Retail Terdekat", value=2)
        new_jumlah_pasar_tradisional = st.number_input("Jumlah Pasar Tradisional", value=1)
        new_jarak_pasar = st.number_input("Jarak ke Pasar Tradisional Terdekat (meter)", value=363.08)
        new_jumlah_restoran = st.number_input("Jumlah Restoran/Rumah Makan Sekitar", value=1)
        new_jumlah_fasilitas_belanja = st.number_input("Jumlah Fasilitas Belanja Komersial", value=11)
        new_jumlah_toko_ponsel = st.number_input("Jumlah Toko Ponsel Sekitar", value=13)
        
        kategori_wilayah = st.selectbox("Kategori Wilayah Kebijakan", ["Perdesaan", "Perkampungan", "Perkotaan"])
        kat_mapped = {"Perdesaan": 0, "Perkampungan": 1, "Perkotaan": 2}[kategori_wilayah]
        
        tipe_jalan = st.selectbox("Jenis Akses Jalan Utama", ["primary", "residential", "tertiary", "secondary", "living_street", "trunk"])
        jalan_mapped = {"primary": 0, "residential": 1, "tertiary": 2, "secondary": 3, "living_street": 4, "trunk": 5}[tipe_jalan]

    submitted = st.form_submit_button("Jalankan Kalkulasi Prediksi", type="primary")

# ========================================================
# 3. PROSES KALKULASI PREDIKSI SAAT TOMBOL DITEKAN
# ========================================================
if submitted:
    # --- Feature Engineering Instan ---
    commercial_hub_index = new_jumlah_restoran + new_jumlah_fasilitas_belanja + new_jumlah_toko_ponsel
    premium_spot_score = new_lebar_ruko * new_umk
    comp_per_pop = new_jumlah_kompetitor / (new_penduduk + 1)
    
    # Membuat dictionary data mentah dari input pengguna
    raw_input_dict = {
        'umk': new_umk, 'penduduk': new_penduduk, 'kemiskinan': new_kemiskinan,
        'jumlah_fasilitas_belanja': new_jumlah_fasilitas_belanja, 'jumlah_toko_ponsel': new_jumlah_toko_ponsel,
        'jumlah_kompetitor': new_jumlah_kompetitor, 'jumlah_pasar_tradisional': new_jumlah_pasar_tradisional,
        'jarak_pasar': new_jarak_pasar, 'jumlah_restoran': new_jumlah_restoran, 'lebar_ruko': new_lebar_ruko,
        'jalan_mapped': jalan_mapped, 'jumlah_bangunan': new_jumlah_bangunan,
        'commercial_hub_index': commercial_hub_index, 'premium_spot_score': premium_spot_score, 'comp_per_pop': comp_per_pop,
        'kategori_wilayah_mapped': kat_mapped
    }
    df_sim = pd.DataFrame([raw_input_dict])

    # --------------------------------------------------------
    # MODEL 1: ORDINARY LEAST SQUARES (OLS) PREDICTION
    # --------------------------------------------------------
    X_ols_raw = df_sim[gwr_meta['features_final']].copy()
    X_ols_scaled = scaler_ols.transform(X_ols_raw) # Menggunakan scaler_ols yang tepat
    X_ols_df = pd.DataFrame(X_ols_scaled, columns=gwr_meta['features_final'])
    X_ols_const = np.insert(X_ols_df.values[0], 0, 1.0) # Menyisipkan konstanta intersep bernilai 1.0
    pred_omzet_ols = np.dot(ols_model.params, X_ols_const)

    # --------------------------------------------------------
    # MODEL 2: RANDOM FOREST REGRESSOR OPTIMIZED
    # --------------------------------------------------------
    df_sim_rf = df_sim.copy()
    # Menggunakan scaler_rf yang tepat untuk fitur dasar
    df_sim_rf[gwr_meta['features_final']] = scaler_rf.transform(df_sim_rf[gwr_meta['features_final']])
    X_rf_final = df_sim_rf[gwr_meta['features_eng']]
    pred_log_rf = rf_model.predict(X_rf_final)[0]
    pred_omzet_rf = np.expm1(pred_log_rf) # Mengembalikan nilai log transformasi ke nilai mata uang Rupiah asli

    # --------------------------------------------------------
    # MODEL 3: GEOGRAPHICALLY WEIGHTED REGRESSION (GWR) SPASIAL
    # --------------------------------------------------------
    # Mengonversi koordinat input bumi (WGS84) ke metrik proyeksi lokal (UTM Zone 48S meteran)
    transformer = Transformer.from_crs("EPSG:4326", "EPSG:32748", always_xy=True)
    x_sim_utm, y_sim_utm = transformer.transform(new_lon, new_lat)
    
    # Mencari indeks lokasi fisik cabang historis terdekat (Spatial Nearest Neighbor)
    coords_sim_utm = np.array([[x_sim_utm, y_sim_utm]])
    coords_train_utm = df_ref[['x_utm', 'y_utm']].values
    distances = cdist(coords_sim_utm, coords_train_utm)
    nearest_idx = np.argmin(distances[0])
    
    cabang_terdekat = df_ref.iloc[nearest_idx]
    jarak_km = distances[0][nearest_idx] / 1000

    # Melakukan Z-Score standarisasi manual menggunakan rata-rata & deviasi standar data latih asli GWR
    X_sim_gwr_raw = df_sim[gwr_meta['features_GW2']].values[0]
    X_sim_gwr_scaled = (X_sim_gwr_raw - gwr_meta['X_gwr_mean']) / gwr_meta['X_gwr_std']
    
    # Mengambil nilai parameter Beta (koefisien lokal) unik dari cabang terdekat tersebut
    local_betas = [cabang_terdekat['gwr_intercept']] + [cabang_terdekat[f'gwr_beta_{col}'] for col in gwr_meta['features_GW2']]
    X_predict_block = np.hstack([1, X_sim_gwr_scaled])
    
    # Melakukan perkalian dot matriks koefisien lokal dengan nilai fitur terstandardisasi
    pred_std_gwr = np.dot(X_predict_block, local_betas)
    
    # Melakukan invers Z-Score ke skala Rupiah sesungguhnya
    pred_omzet_gwr = (pred_std_gwr * gwr_meta['y_gwr_std']) + gwr_meta['y_gwr_mean']

    # ========================================================
    # 4. VISUALISASI HASIL AKHIR (UI DASHBOARD CARDS)
    # ========================================================
    st.markdown("---")
    st.subheader("Hasil Estimasi Komparasi Omzet Bulanan")
    
    m_col1, m_col2, m_col3 = st.columns(3)
    with m_col1:
        st.metric(label="1. OLS Baseline Model", value=f"Rp {max(0.0, pred_omzet_ols):,.2f}".replace(",", "."))
        st.caption("Prediksi tren linear global tanpa pembobotan lokal geografis.")
    with m_col2:
        st.metric(label="2. Random Forest (Optimasi)", value=f"Rp {pred_omzet_rf:,.2f}".replace(",", "."))
        st.caption("Akurat dalam mengenali pola interaksi non-linear parameter bisnis.")
    with m_col3:
        st.metric(label="3. Spasial GWR Hasil Koreksi", value=f"Rp {pred_omzet_gwr:,.2f}".replace(",", "."))
        st.caption(f"Berbasis bobot geografis lokal dari titik terdekat: **{cabang_terdekat['nama_cabang']}** (Jarak: {jarak_km:.2f} km).")

    st.info("**Petunjuk Analisis Manajemen:** Model GWR memberikan estimasi terbaik berdasarkan realita kedekatan fisik wilayah retail, sementara Random Forest unggul dalam ketajaman logika matematis kombinasi variabel.")
