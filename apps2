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
    rf_model, scaler_rf, model_ols, scaler_ols, gwr_meta = load_all_models()
    df_ref = gwr_meta['df_spatial_reference']
except Exception as e:
    st.error(f"Gagal memuat komponen model (.pkl). File pkl sudah di-upload di root repository GitHub. Detail Error: {e}")
    st.stop()


# ========================================================
# FUNGSIONALITAS UTAMA PREDIKSI (Fungsi Helper untuk Batch/Single)
# ========================================================
def hitung_prediksi_batch(df_input):
    """
    Menerima DataFrame input mentah dengan kolom-kolom yang sesuai,
    mengembalikan DataFrame yang sudah terisi hasil prediksi OLS, RF, dan GWR.
    """
    df_res = df_input.copy()
    
    # Menerapkan mapping kategori jika masih berupa teks
    kat_map = {"Perdesaan": 0, "Perkampungan": 1, "Perkotaan": 2}
    jalan_map = {"primary": 0, "residential": 1, "tertiary": 2, "secondary": 3, "living_street": 4, "trunk": 5}
    
    if 'kategori_wilayah' in df_res.columns:
        df_res['kategori_wilayah_mapped'] = df_res['kategori_wilayah'].map(kat_map).fillna(0).astype(int)
    if 'tipe_jalan' in df_res.columns:
        df_res['jalan_mapped'] = df_res['tipe_jalan'].map(jalan_map).fillna(1).astype(int)

    # --- Feature Engineering ---
    df_res['commercial_hub_index'] = df_res['jumlah_restoran'] + df_res['jumlah_fasilitas_belanja'] + df_res['jumlah_toko_ponsel']
    df_res['premium_spot_score'] = df_res['lebar_ruko'] * df_res['umk']
    df_res['comp_per_pop'] = df_res['jumlah_kompetitor'] / (df_res['penduduk'] + 1)
    
    # 1. PREDIKSI MODEL OLS
    X_ols_raw = df_res[gwr_meta['features_final']].copy()
    X_ols_scaled = scaler_ols.transform(X_ols_raw)
    X_ols_df = pd.DataFrame(X_ols_scaled, columns=gwr_meta['features_final'])
    X_ols_const = np.hstack([np.ones((len(X_ols_df), 1)), X_ols_df.values])
    preds_ols = np.dot(X_ols_const, ols_model.params)
    df_res['Prediksi_Omzet_OLS'] = np.clip(preds_ols, 0, None)

    # 2. PREDIKSI MODEL RANDOM FOREST
    df_sim_rf = df_res.copy()
    df_sim_rf[gwr_meta['features_final']] = scaler_rf.transform(df_sim_rf[gwr_meta['features_final']])
    X_rf_final = df_sim_rf[gwr_meta['features_eng']]
    pred_log_rf = rf_model.predict(X_rf_final)
    df_res['Prediksi_Omzet_RF'] = np.expm1(pred_log_rf)

    # 3. PREDIKSI MODEL SPASIAL GWR
    transformer = Transformer.from_crs("EPSG:4326", "EPSG:32748", always_xy=True)
    preds_gwr = []
    nama_cabang_dekat = []
    jarak_dekat_km = []
    
    coords_train_utm = df_ref[['x_utm', 'y_utm']].values

    for idx, row in df_res.iterrows():
        x_sim_utm, y_sim_utm = transformer.transform(row['longitude'], row['latitude'])
        coords_sim_utm = np.array([[x_sim_utm, y_sim_utm]])
        
        distances = cdist(coords_sim_utm, coords_train_utm)
        nearest_idx = np.argmin(distances[0])
        
        cabang_terdekat = df_ref.iloc[nearest_idx]
        jarak_km = distances[0][nearest_idx] / 1000
        
        nama_cabang_dekat.append(cabang_terdekat['nama_cabang'])
        jarak_dekat_km.append(jarak_km)

        X_sim_gwr_raw = row[gwr_meta['features_GW2']].values
        X_sim_gwr_scaled = (X_sim_gwr_raw - gwr_meta['X_gwr_mean']) / gwr_meta['X_gwr_std']
        
        local_betas = [cabang_terdekat['gwr_intercept']] + [cabang_terdekat[f'gwr_beta_{col}'] for col in gwr_meta['features_GW2']]
        X_predict_block = np.hstack([1, X_sim_gwr_scaled])
        
        pred_std_gwr = np.dot(X_predict_block, local_betas)
        pred_omzet_gwr = (pred_std_gwr * gwr_meta['y_gwr_std']) + gwr_meta['y_gwr_mean']
        preds_gwr.append(pred_omzet_gwr)

    df_res['Prediksi_Omzet_GWR'] = preds_gwr
    df_res['Cabang_Terdekat_Ref'] = nama_cabang_dekat
    df_res['Jarak_Ref_KM'] = jarak_dekat_km
    
    return df_res


# ========================================================
# 2. NAVIGASI UTAMA (SIDEBAR MENU)
# ========================================================
st.sidebar.title("Navigasi Aplikasi")
menu_terpilih = st.sidebar.radio(
    "Pilih Menu Halaman:",
    ["Simulasi Cabang Baru", "Performa & Evaluasi Model"]
)

st.sidebar.markdown("---")
st.sidebar.caption("Data Analytics © 2026")


# ========================================================
# HALAMAN 1: SIMULASI CABANG BARU (BISA SINGLE / BATCH)
# ========================================================
if menu_terpilih == "Simulasi Cabang Baru":
    st.title("Aplikasi Simulasi Perbandingan 3 Model Forecasting Omzet PGI")
    st.write("Sistem ini membandingkan hasil model OLS, Random Forest, dan Koreksi Spasial GWR untuk membantu analisis ekspansi cabang.")

    # Pilihan Metode Input
    metode_input = st.radio("Pilih Metode Input Data:", ["Upload File Batch (Banyak Data)", "Input Manual (Satu data)"], horizontal=True)

    if metode_input == "Upload File Batch (Banyak Data)":
        st.markdown("### Upload File Excel atau CSV")
        st.write("Pastikan file Anda memiliki kolom-kolom berikut: `latitude`, `longitude`, `umk`, `penduduk`, `kemiskinan`, `lebar_ruko`, `jumlah_bangunan`, `jumlah_kompetitor`, `jumlah_pasar_tradisional`, `jarak_pasar`, `jumlah_restoran`, `jumlah_fasilitas_belanja`, `jumlah_toko_ponsel`, `kategori_wilayah`, `tipe_jalan`")
        
        # Contoh Template Unduhan
        template_data = {
            'id_cabang_rencana': ['CABANG_A', 'CABANG_B'], 'latitude': [-6.925914, -6.2088], 'longitude': [107.588618, 106.8456],
            'umk': [4482914, 5000000], 'penduduk': [94158, 120000], 'kemiskinan': [0.04, 0.02], 'lebar_ruko': [450, 500], 'jumlah_bangunan': [18840, 25000],
            'jumlah_kompetitor': [2, 5], 'jumlah_pasar_tradisional': [1, 2], 'jarak_pasar': [363.08, 500.0], 'jumlah_restoran': [1, 10],
            'jumlah_fasilitas_belanja': [11, 20], 'jumlah_toko_ponsel': [13, 15], 'kategori_wilayah': ['Perkotaan', 'Perkotaan'], 'tipe_jalan': ['secondary', 'primary']
        }
        df_template = pd.DataFrame(template_data)
        
        st.download_button(
            label="📥 Unduh Contoh Template Excel",
            data=df_template.to_csv(index=False).encode('utf-8'),
            file_name='template_simulasi_pgi.csv',
            mime='text/csv'
        )

        file_diunggah = st.file_uploader("Pilih file CSV atau Excel:", type=['csv', 'xlsx'])
        
        if file_diunggah is not None:
            try:
                if file_diunggah.name.endswith('.csv'):
                    df_batch = pd.read_csv(file_diunggah)
                else:
                    df_batch = pd.read_excel(file_diunggah)
                
                st.success("File berhasil dimuat! Menampilkan 5 data pertama:")
                st.dataframe(df_batch.head())
                
                tombol_proses = st.button("Jalankan Prediksi Massal", type="primary")
                
                if tombol_proses:
                    with st.spinner("Sedang memproses seluruh data dengan 3 model..."):
                        df_hasil = hitung_prediksi_batch(df_batch)
                    
                    st.markdown("---")
                    st.subheader("📊 Hasil Prediksi Batch")
                    
                    # Kolom yang ingin ditampilkan secara ringkas di preview
                    kolom_tampil = []
                    if 'id_cabang_rencana' in df_hasil.columns: kolom_tampil.append('id_cabang_rencana')
                    kolom_tampil.extend(['Prediksi_Omzet_OLS', 'Prediksi_Omzet_RF', 'Prediksi_Omzet_GWR', 'Cabang_Terdekat_Ref', 'Jarak_Ref_KM'])
                    
                    st.dataframe(
                        df_hasil[kolom_tampil].style.format({
                            'Prediksi_Omzet_OLS': 'Rp {:,.2f}',
                            'Prediksi_Omzet_RF': 'Rp {:,.2f}',
                            'Prediksi_Omzet_GWR': 'Rp {:,.2f}',
                            'Jarak_Ref_KM': '{:.2f} Km'
                        }), use_container_width=True
                    )
                    
                    # Download link hasil lengkap
                    st.download_button(
                        label="📥 Unduh Hasil Prediksi Lengkap (.CSV)",
                        data=df_hasil.to_csv(index=False).encode('utf-8'),
                        file_name='hasil_prediksi_3model_pgi.csv',
                        mime='text/csv'
                    )
            except Exception as err:
                st.error(f"Gagal memproses file. Pastikan format kolom sesuai template. Detail Error: {err}")

    else:
        # --- INPUT MANUAL (LOGIKA LAMA ANDA) ---
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
                tipe_jalan = st.selectbox("Jenis Akses Jalan Utama", ["primary", "residential", "tertiary", "secondary", "living_street", "trunk"])

            submitted = st.form_submit_button("Jalankan Kalkulasi Prediksi", type="primary")

        if submitted:
            # Memasukkan input manual ke format DataFrame agar bisa dibaca fungsi batch
            df_manual_input = pd.DataFrame([{
                'latitude': new_lat, 'longitude': new_lon, 'umk': new_umk, 'penduduk': new_penduduk,
                'kemiskinan': new_kemiskinan, 'lebar_ruko': new_lebar_ruko, 'jumlah_bangunan': new_jumlah_bangunan,
                'jumlah_kompetitor': new_jumlah_kompetitor, 'jumlah_pasar_tradisional': new_jumlah_pasar_tradisional,
                'jarak_pasar': new_jarak_pasar, 'jumlah_restoran': new_jumlah_restoran,
                'jumlah_fasilitas_belanja': new_jumlah_fasilitas_belanja, 'jumlah_toko_ponsel': new_jumlah_toko_ponsel,
                'kategori_wilayah': kategori_wilayah, 'tipe_jalan': tipe_jalan
            }])
            
            df_hasil_manual = hitung_prediksi_batch(df_manual_input)
            row_hasil = df_hasil_manual.iloc[0]

            # VISUALISASI HASIL AKHIR
            st.markdown("---")
            st.subheader("Hasil Estimasi Komparasi Omzet Bulanan")
            
            m_col1, m_col2, m_col3 = st.columns(3)
            with m_col1:
                st.metric(label="1. OLS Baseline Model", value=f"Rp {row_hasil['Prediksi_Omzet_OLS']:,.2f}".replace(",", "."))
                st.caption("Prediksi tren linear global tanpa pembobotan lokal geografis.")
            with m_col2:
                st.metric(label="2. Random Forest (Optimasi)", value=f"Rp {row_hasil['Prediksi_Omzet_RF']:,.2f}".replace(",", "."))
                st.caption("Akurat dalam mengenali pola interaksi non-linear parameter bisnis.")
            with m_col3:
                st.metric(label="3. Spasial GWR Hasil Koreksi", value=f"Rp {row_hasil['Prediksi_Omzet_GWR']:,.2f}".replace(",", "."))
                st.caption(f"Berbasis bobot geografis lokal dari titik terdekat: **{row_hasil['Cabang_Terdekat_Ref']}** (Jarak: {row_hasil['Jarak_Ref_KM']:.2f} km).")

            st.info("**Petunjuk Analisis Manajemen:** Model GWR memberikan estimasi terbaik berdasarkan realita kedekatan fisik wilayah retail, sementara Random Forest unggul dalam ketajaman logika matematis kombinasi variabel.")


# ========================================================
# HALAMAN 2: PERFORMA & EVALUASI MODEL (KODE TETAP)
# ========================================================
elif menu_terpilih == "Performa & Evaluasi Model":
    # (Bagian evaluasi model di bawah ini tetap sama seperti script asli Anda)
    st.title("Laporan Metrik Performa & Evaluasi Pemodelan")
    st.write("Halaman ini menyajikan validasi hasil evaluasi matematis dan statistik untuk ketiga arsitektur model berdasarkan data historis.")
    
    tab_ols, tab_rf, tab_gwr = st.tabs([" 1. Baseline MLR (OLS)", " 2. Random Forest Regressor", " 3. Geographically Weighted Regression (GWR)"])
    
    with tab_ols:
        st.header("Multiple Linear Regression - OLS Baseline")
        c1, c2, c3 = st.columns(3)
        c1.metric(label="R-squared (R²)", value="0.2179")
        c2.metric(label="MAE", value="Rp 160.364.206,25")
        st.warning("**Catatan Evaluasi:** Nilai R² yang cenderung rendah (21,79%) membuktikan hubungan faktor pembuat omzet antar wilayah bersifat non-linear dan heterogen spasi.")

    with tab_rf:
        st.header("Random Forest Regressor (Optimized)")
        rf1, rf2 = st.columns(2)
        rf1.metric(label="Final Optimized R-squared", value="0.2675")
        rf2.metric(label="Mean Absolute Error (MAE)", value="Rp 154.615.822,10")
        
        fi_data = {
            'Feature / Variabel Indikator': ['umk', 'jumlah_bangunan', 'lebar_ruko', 'penduduk', 'jumlah_toko_ponsel', 'jumlah_kompetitor', 'jumlah_fasilitas_belanja', 'jarak_pasar', 'kemiskinan', 'jalan_mapped', 'jumlah_restoran', 'jumlah_pasar_tradisional'],
            'Importance Score': [0.161026, 0.136584, 0.109638, 0.106220, 0.099606, 0.092284, 0.085335, 0.077356, 0.049833, 0.033503, 0.025338, 0.023277]
        }
        df_fi = pd.DataFrame(fi_data)
        col_table, col_chart = st.columns([2, 3])
        with col_table: st.dataframe(df_fi.style.format({'Importance Score': '{:.4f}'}), use_container_width=True)
        with col_chart: st.bar_chart(data=df_fi, x='Feature / Variabel Indikator', y='Importance Score', horizontal=True)

    with tab_gwr:
        st.header("Geographically Weighted Regression (GWR) Spasial")
        g1, g2, g3, g4 = st.columns(4)
        g1.metric(label="R-squared Global (R²)", value="0.3480")
        g2.metric(label="Adjusted R-squared", value="0.2940")
        g3.metric(label="AICc", value="1.975,26")
        g4.metric(label="RMSE Spasial", value="Rp 207.524.678,53")
        
        coef_data = {
            'Nama Parameter/Variabel': ['Intercept (Konstanta Spasial)', 'umk', 'penduduk', 'kemiskinan', 'jumlah_kompetitor', 'jumlah_pasar_tradisional', 'jarak_pasar', 'lebar_ruko', 'jumlah_bangunan', 'kategori_wilayah_mapped', 'jalan_mapped', 'commercial_hub_index', 'premium_spot_score', 'comp_per_pop'],
            'Rata-rata Koefisien (Beta)': [-0.1700, 0.3753, -0.0221, 0.0065, 0.1421, 0.0439, 0.0633, 0.2014, -0.0652, 0.0816, -0.0864, 0.2125, -0.0562, -1.3624]
        }
        df_coef = pd.DataFrame(coef_data)
        st.table(df_coef)
