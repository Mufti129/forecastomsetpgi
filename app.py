import streamlit as st
import pandas as pd
import numpy as np
import joblib
from scipy.spatial.distance import cdist
from pyproj import Transformer

st.set_page_config(page_title="PGI 3-Model Forecasting", layout="wide")

st.title("📊 Simulasi Perbandingan 3 Model Forecasting Omzet PGI")
st.write("Masukkan indikator calon lokasi cabang baru untuk melihat perbandingan prediksi dari model OLS, Random Forest, dan GWR Spasial.")

# 1. Load Assets
@st.cache_resource
def load_assets():
    rf_model = joblib.load('rf_model.pkl')
    ols_model = joblib.load('ols_model.pkl')
    scaler = joblib.load('scaler.pkl')
    df_ref = joblib.load('df_spatial_reference.pkl')
    return rf_model, ols_model, scaler, df_ref

rf_model, ols_model, scaler, df_ref = load_assets()

# 2. Layout Form Input (Dibagi menjadi 2 Kolom Besar)
st.markdown("### 📝 Input Parameter Cabang Baru")
with st.form("input_form"):
    col1, col2 = st.columns(2)
    
    with col1:
        new_lat = st.number_input("Latitude", value=-6.925914, format="%.6f")
        new_lon = st.number_input("Longitude", value=107.588618, format="%.6f")
        new_umk = st.number_input("UMK Wilayah (Rp)", value=4482914)
        new_penduduk = st.number_input("Jumlah Penduduk", value=94158)
        new_kemiskinan = st.slider("Proporsi Kemiskinan", 0.0, 1.0, 0.04)
        new_lebar_ruko = st.number_input("Lebar Ruko (cm)", value=450)
        new_jumlah_bangunan = st.number_input("Jumlah Bangunan Sekitar", value=18840)

    with col2:
        new_jumlah_kompetitor = st.number_input("Jumlah Kompetitor Terdekat", value=2)
        new_jumlah_pasar_tradisional = st.number_input("Jumlah Pasar Tradisional", value=1)
        new_jarak_pasar = st.number_input("Jarak ke Pasar (meter)", value=363)
        new_jumlah_restoran = st.number_input("Jumlah Restoran", value=1)
        new_jumlah_fasilitas_belanja = st.number_input("Fasilitas Belanja", value=11)
        new_jumlah_toko_ponsel = st.number_input("Toko Ponsel", value=13)
        
        kategori_wilayah = st.selectbox("Kategori Wilayah", ["Perdesaan", "Perkampungan", "Perkotaan"])
        kategori_wilayah_mapped = {"Perdesaan": 0, "Perkampungan": 1, "Perkotaan": 2}[kategori_wilayah]
        
        tipe_jalan = st.selectbox("Tipe Jalan", ["primary", "residential", "tertiary", "secondary", "living_street", "trunk"])
        jalan_mapped = {"primary": 0, "residential": 1, "tertiary": 2, "secondary": 3, "living_street": 4, "trunk": 5}[tipe_jalan]

    submitted = st.form_submit_button("🚀 Jalankan 3 Model Prediksi", type="primary")

# 3. Eksekusi Prediksi Saat Tombol Ditekan
if submitted:
    # --- Feature Engineering Instan ---
    commercial_hub_index = new_jumlah_restoran + new_jumlah_fasilitas_belanja + new_jumlah_toko_ponsel
    premium_spot_score = new_lebar_ruko * new_umk
    comp_per_pop = new_jumlah_kompetitor / (new_penduduk + 1)
    
    # Susun Array Fitur Utama
    features_list = [
        new_umk, new_penduduk, new_kemiskinan, new_jumlah_fasilitas_belanja, new_jumlah_toko_ponsel,
        new_jumlah_kompetitor, new_jumlah_pasar_tradisional, new_jarak_pasar, new_jumlah_restoran,
        new_lebar_ruko, jalan_mapped, new_jumlah_bangunan, commercial_hub_index,
        premium_spot_score, comp_per_pop
    ]
    
    features_eng = [
        'umk','penduduk','kemiskinan','jumlah_fasilitas_belanja','jumlah_toko_ponsel',
        'jumlah_kompetitor','jumlah_pasar_tradisional','jarak_pasar','jumlah_restoran',
        'lebar_ruko','jalan_mapped','jumlah_bangunan','commercial_hub_index',
        'premium_spot_score','comp_per_pop'
    ]
    
    input_data = pd.DataFrame([features_list], columns=features_eng)
    input_scaled = scaler.transform(input_data)
    
    # --- MODEL 1: Prediksi via Random Forest ---
    pred_log_rf = rf_model.predict(input_scaled)[0]
    pred_omzet_rf = np.expm1(pred_log_rf)
    
    # --- MODEL 2: Prediksi via OLS (Linear Regression) ---
    input_scaled_ols = np.insert(input_scaled[0], 0, 1) # tambah konstanta 1 di depan
    pred_log_ols = np.dot(ols_model.params, input_scaled_ols)
    pred_omzet_ols = np.expm1(pred_log_ols)
    
    # --- MODEL 3: Prediksi via GWR (Geographically Weighted Regression) ---
    # Konversi koordinat input ke UTM
    transformer = Transformer.from_crs("EPSG:4326", "EPSG:32748", always_xy=True)
    x_sim_utm, y_sim_utm = transformer.transform(new_lon, new_lat)
    
    # Cari Cabang Historis Terdekat (Spatial Nearest Neighbor)
    coords_sim = np.array([[x_sim_utm, y_sim_utm]])
    coords_train = df_ref[['x_utm', 'y_utm']].values
    distances = cdist(coords_sim, coords_train)
    nearest_idx = np.argmin(distances[0])
    
    # Ambil nilai parameter lokal (Beta) dari cabang tetangga terdekat tersebut
    cabang_tetangga = df_ref.iloc[nearest_idx]
    gwr_intercept = cabang_tetangga['gwr_intercept']
    
    # Hitung nilai prediksi GWR menggunakan koefisien spasial lokal
    gwr_prediction = gwr_intercept
    for col in features_eng:
        beta_lokal = cabang_tetangga[f'gwr_beta_{col}']
        val_fitur = input_data[col].values[0] # GWR menggunakan nilai asli non-scaled di skrip asli
        gwr_prediction += beta_lokal * val_fitur
        
    pred_omzet_gwr = gwr_prediction # GWR di skrip langsung menggunakan target asli (bukan log)

    # --- TAMPILKAN HASIL PERBANDINGAN DI UI ---
    st.markdown("---")
    st.subheader("📌 Perbandingan Hasil Estimasi Omzet Bulanan")
    
    m_col1, m_col2, m_col3 = st.columns(3)
    
    with m_col1:
        st.metric(
            label="1. Baseline Model (OLS)", 
            value=f"Rp {pred_omzet_ols:,.2f}".replace(",", "."),
            help="Pendekatan Statistik Regresi Linier Global"
        )
        st.caption("Cocok untuk melihat tren makro linear, namun kurang sensitif pada variasi lokal ekstrem.")
        
    with m_col2:
        st.metric(
            label="2. Machine Learning (Random Forest)", 
            value=f"Rp {pred_omzet_rf:,.2f}".replace(",", "."),
            help="Pendekatan Non-Linear dengan Transformasi Log"
        )
        st.caption("Paling stabil secara global dalam menangani interaksi fitur yang rumit.")
        
    with m_col3:
        st.metric(
            label="3. Analisis Spasial (GWR Lokal)", 
            value=f"Rp {pred_omzet_gwr:,.2f}".replace(",", "."),
            help="Pendekatan Bobot Geografis Berbasis Cabang Terdekat"
        )
        st.caption(f"Sangat sensitif terhadap lokasi. Menggunakan karakteristik lokal dari cabang terdekat: **{cabang_tetangga['nama_cabang']}**.")

    st.success(f"💡 **Rekomendasi Analisis:** Bandingkan nilai **Random Forest** (Akurasi Data) dengan **GWR** (Akurasi Geografis). Jika jarak ke cabang terdekat sangat dekat, nilai GWR cenderung memberikan gambaran riil kondisi lapangan di klaster wilayah tersebut.")
