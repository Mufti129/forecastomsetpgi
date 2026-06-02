import streamlit as st
import pandas as pd
import numpy as np
import joblib
from scipy.spatial.distance import cdist
from pyproj import Transformer
import os

st.set_page_config(page_title="PGI 3-Model Analytics & Forecasting", layout="wide")

# ========================================================
# 1. LOAD OBJEK MODEL DENGAN CACHING AMAN
# ========================================================
@st.cache_resource
def load_all_models():
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
# 2. NAVIGASI UTAMA (SIDEBAR MENU)
# ========================================================
st.sidebar.title("Navigasi Aplikasi")
menu_terpilih = st.sidebar.radio(
    "Pilih Menu Halaman:",
    ["Simulasi Cabang Baru", "Performa & Evaluasi Model"]
)

st.sidebar.markdown("---")
st.sidebar.caption("Sistem Forecasting Omzet Cabang PGI © 2026")


# ========================================================
# HALAMAN 1: SIMULASI CABANG BARU
# ========================================================
if menu_terpilih == "Simulasi Cabang Baru":
    st.title("Aplikasi Simulasi Perbandingan 3 Model Forecasting Omzet PGI")
    st.write("Sistem ini membandingkan hasil model OLS, Random Forest, dan Koreksi Spasial GWR secara instan untuk membantu analisis ekspansi cabang.")

    st.markdown("### Masukkan Parameter Karakteristik Cabang Baru")
    with st.form("simulation_form"):
        col1, col2 = st.columns(2)
        with col1:
            new_lat = st.number_input("Latitude (Garis Lintang)", value=-6.925914, format="%.6f")
            new_lon = st.number_input("Longitude (Garis Buku)", value=107.588618, format="%.6f")
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

    if submitted:
        # --- Feature Engineering Instan ---
        commercial_hub_index = new_jumlah_restoran + new_jumlah_fasilitas_belanja + new_jumlah_toko_ponsel
        premium_spot_score = new_lebar_ruko * new_umk
        comp_per_pop = new_jumlah_kompetitor / (new_penduduk + 1)
        
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

        # MODEL 1: OLS
        X_ols_raw = df_sim[gwr_meta['features_final']].copy()
        X_ols_scaled = scaler_ols.transform(X_ols_raw)
        X_ols_df = pd.DataFrame(X_ols_scaled, columns=gwr_meta['features_final'])
        X_ols_const = np.insert(X_ols_df.values[0], 0, 1.0)
        pred_omzet_ols = np.dot(ols_model.params, X_ols_const)

        # MODEL 2: RANDOM FOREST
        df_sim_rf = df_sim.copy()
        df_sim_rf[gwr_meta['features_final']] = scaler_rf.transform(df_sim_rf[gwr_meta['features_final']])
        X_rf_final = df_sim_rf[gwr_meta['features_eng']]
        pred_log_rf = rf_model.predict(X_rf_final)[0]
        pred_omzet_rf = np.expm1(pred_log_rf)

        # MODEL 3: GWR SPASIAL
        transformer = Transformer.from_crs("EPSG:4326", "EPSG:32748", always_xy=True)
        x_sim_utm, y_sim_utm = transformer.transform(new_lon, new_lat)
        
        coords_sim_utm = np.array([[x_sim_utm, y_sim_utm]])
        coords_train_utm = df_ref[['x_utm', 'y_utm']].values
        distances = cdist(coords_sim_utm, coords_train_utm)
        nearest_idx = np.argmin(distances[0])
        
        cabang_terdekat = df_ref.iloc[nearest_idx]
        jarak_km = distances[0][nearest_idx] / 1000

        X_sim_gwr_raw = df_sim[gwr_meta['features_GW2']].values[0]
        X_sim_gwr_scaled = (X_sim_gwr_raw - gwr_meta['X_gwr_mean']) / gwr_meta['X_gwr_std']
        
        local_betas = [cabang_terdekat['gwr_intercept']] + [cabang_terdekat[f'gwr_beta_{col}'] for col in gwr_meta['features_GW2']]
        X_predict_block = np.hstack([1, X_sim_gwr_scaled])
        
        pred_std_gwr = np.dot(X_predict_block, local_betas)
        pred_omzet_gwr = (pred_std_gwr * gwr_meta['y_gwr_std']) + gwr_meta['y_gwr_mean']

        # VISUALISASI HASIL AKHIR
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


# ========================================================
# HALAMAN 2: PERFORMA & EVALUASI MODEL (MENU BARU)
# ========================================================
elif menu_terpilih == "Performa & Evaluasi Model":
    st.title("Laporan Metrik Performa & Evaluasi Pemodelan")
    st.write("Halaman ini menyajikan validasi hasil evaluasi matematis dan statistik untuk ketiga arsitektur model berdasarkan data historis.")
    
    # Membuat 3 Tabs untuk masing-masing model agar rapi
    tab_ols, tab_rf, tab_gwr = st.tabs([
        " 1. Baseline MLR (OLS)", 
        " 2. Random Forest Regressor", 
        " 3. Geographically Weighted Regression (GWR)"
    ])
    
    # --- TAB OLS ---
    with tab_ols:
        st.header("Multiple Linear Regression - OLS Baseline")
        st.write("Model global konvensional untuk mengukur pengaruh linier homogen di seluruh cabang.")
        
        c1, c2, c3 = st.columns(3)
        c1.metric(label="R-squared (R²)", value="0.2179")
        c2.metric(label="MAE", value="Rp 160.364.206,25")
        c3.metric(label="RMSE", value="Rp 246.962.146,72")
        
        st.warning("**Catatan Evaluasi:** Nilai R² yang cenderung rendah (21,79%) membuktikan hubungan faktor pembuat omzet antar wilayah bersifat non-linear dan heterogen spasi, sehingga kurang disarankan sebagai acuan tunggal.")

    # --- TAB RANDOM FOREST ---
    with tab_rf:
        st.header("Random Forest Regressor (Optimized)")
        st.write("Pendekatan Machine Learning berbasis pohon keputusan non-linear dengan optimasi log-transform target.")
        
        rf1, rf2 = st.columns(2)
        rf1.metric(label="Final Optimized R-squared", value="0.2675")
        rf2.metric(label="Mean Absolute Error (MAE)", value="Rp 154.615.822,10")
        
        st.subheader("Variabel Paling Berpengaruh (Feature Importance)")
        st.write("Menunjukkan tingkat kontribusi bobot dari setiap indikator dalam membentuk keputusan prediksi.")
        
        # Menyusun data Importance dari riset Anda ke DataFrame
        fi_data = {
            'Feature / Variabel Indikator': [
                'umk', 'jumlah_bangunan', 'lebar_ruko', 'penduduk', 
                'jumlah_toko_ponsel', 'jumlah_kompetitor', 'jumlah_fasilitas_belanja', 
                'jarak_pasar', 'kemiskinan', 'jalan_mapped', 'jumlah_restoran', 'jumlah_pasar_tradisional'
            ],
            'Importance Score': [
                0.161026, 0.136584, 0.109638, 0.106220, 
                0.099606, 0.092284, 0.085335, 0.077356, 
                0.049833, 0.033503, 0.025338, 0.023277
            ]
        }
        df_fi = pd.DataFrame(fi_data)
        
        # Tampilkan tabel dan chart horizontal sederhana di Streamlit
        col_table, col_chart = st.columns([2, 3])
        with col_table:
            st.dataframe(df_fi.style.format({'Importance Score': '{:.4f}'}), use_container_width=True)
        with col_chart:
            st.bar_chart(data=df_fi, x='Feature / Variabel Indikator', y='Importance Score', horizontal=True)

    # --- TAB GWR ---
    with tab_gwr:
        st.header("Geographically Weighted Regression (GWR) Spasial")
        st.write("Model tingkat lanjut berbasis koordinat proyeksi bumi yang menghitung parameter lokal secara spesifik di tiap wilayah.")
        
        g1, g2, g3, g4 = st.columns(4)
        g1.metric(label="R-squared Global (R²)", value="0.3480")
        g2.metric(label="Adjusted R-squared", value="0.2940")
        g3.metric(label="AICc", value="1.975,26")
        g4.metric(label="RMSE Spasial", value="Rp 207.524.678,53")
        
        st.success("**Kesimpulan Performa Terbaik:** GWR menghasilkan peningkatan akurasi tertinggi dengan nilai **R² mencapai 34,80%**. Hal ini menunjukkan bahwa faktor lokasi geografis memegang peran sangat penting dalam akurasi bisnis retail.")
        
        st.markdown("#### Interpretasi Hasil Analisis Spasial")
        st.info("Sistem mendeteksi adanya **766 set koefisien lokal unik** untuk setiap lokasi cabang historis yang diteliti. Artinya, setiap daerah memiliki sensitivitas pengaruh variabel yang berbeda-beda.")
        
        st.markdown("#### Rata-rata Nilai Koefisien Lokal (Beta Intercept):")
        
        coef_data = {
            'Nama Parameter/Variabel': [
                'Intercept (Konstanta Spasial)', 'umk', 'penduduk', 'kemiskinan', 
                'jumlah_kompetitor', 'jumlah_pasar_tradisional', 'jarak_pasar', 
                'lebar_ruko', 'jumlah_bangunan', 'kategori_wilayah_mapped', 
                'jalan_mapped', 'commercial_hub_index', 'premium_spot_score', 'comp_per_pop'
            ],
            'Rata-rata Koefisien (Beta)': [
                -0.1700, 0.3753, -0.0221, 0.0065, 
                0.1421, 0.0439, 0.0633, 0.2014, 
                -0.0652, 0.0816, -0.0864, 0.2125, 
                -0.0562, -1.3624
            ]
        }
        df_coef = pd.DataFrame(coef_data)
        st.table(df_coef)
