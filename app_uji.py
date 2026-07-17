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
    st.error(f"Gagal memuat komponen model (.pkl). Pastikan semua file pkl ada di root folder repository Anda. Detail Error: {e}")
    st.stop()

# ========================================================
# 2. FUNGSIONALITAS UTAMA PREDIKSI BATCH & MANUAL
# ========================================================
def hitung_prediksi_batch(df_input, ols_model, scaler_ols, rf_model, scaler_rf, gwr_meta, df_ref):
    df_res = df_input.copy()
    
    # --- 1. Penyelarasan / Aliansi Nama Kolom ---
    if 'lat' in df_res.columns and 'latitude' not in df_res.columns:
        df_res['latitude'] = df_res['lat']
    if 'lon' in df_res.columns and 'longitude' not in df_res.columns:
        df_res['longitude'] = df_res['lon']
    if 'jalan' in df_res.columns and 'tipe_jalan' not in df_res.columns:
        df_res['tipe_jalan'] = df_res['jalan']

    # --- 2. Validasi & Pengisian Default Nilai Kosong/Missing ---
    # Mendefinisikan kolom wajib numerik agar tidak crash saat kalkulasi
    kolom_numerik_wajib = [
        'latitude', 'longitude', 'umk', 'penduduk', 'lebar_ruko', 
        'jumlah_kompetitor', 'jumlah_restoran', 'jumlah_fasilitas_belanja', 
        'jumlah_toko_ponsel', 'kemiskinan', 'jumlah_bangunan', 
        'jumlah_pasar_tradisional', 'jarak_pasar'
    ]
    
    for col in kolom_numerik_wajib:
        if col not in df_res.columns:
            df_res[col] = 0.0  # Set default 0 jika kolom tidak ada di file upload
        else:
            df_res[col] = pd.to_numeric(df_res[col], errors='coerce').fillna(0.0)

    # Pastikan koordinat aman dan tidak bernilai NaN (menggunakan default Jakarta/Bandung jika kosong)
    df_res['latitude'] = df_res['latitude'].replace(0.0, -6.925914)
    df_res['longitude'] = df_res['longitude'].replace(0.0, 107.588618)

    # --- 3. Mapping Kategori & Jalan ke Bentuk Numerik ---
    kat_map = {"Perdesaan": 0, "Perkampungan": 1, "Perkotaan": 2}
    jalan_map = {"primary": 0, "residential": 1, "tertiary": 2, "secondary": 3, "living_street": 4, "trunk": 5}
    
    if 'kategori_wilayah' not in df_res.columns:
        df_res['kategori_wilayah'] = 'Perdesaan'
    if 'tipe_jalan' not in df_res.columns:
        df_res['tipe_jalan'] = 'residential'

    df_res['kategori_wilayah_mapped'] = df_res['kategori_wilayah'].map(kat_map).fillna(0).astype(int)
    df_res['jalan_mapped'] = df_res['tipe_jalan'].map(jalan_map).fillna(1).astype(int)

    # --- 4. Feature Engineering ---
    df_res['commercial_hub_index'] = df_res['jumlah_restoran'] + df_res['jumlah_fasilitas_belanja'] + df_res['jumlah_toko_ponsel']
    df_res['premium_spot_score'] = df_res['lebar_ruko'] * df_res['umk']
    df_res['comp_per_pop'] = df_res['jumlah_kompetitor'] / (df_res['penduduk'] + 1)

    # --- 5. PREDIKSI MODEL OLS ---
    X_ols_raw = df_res[gwr_meta['features_final']].copy()
    X_ols_scaled = scaler_ols.transform(X_ols_raw)
    X_ols_df = pd.DataFrame(X_ols_scaled, columns=gwr_meta['features_final'])
    X_ols_const = np.hstack([np.ones((len(X_ols_df), 1)), X_ols_df.values])
    preds_ols = ols_model.predict(X_ols_const)
    df_res['Prediksi_Omzet_OLS'] = np.clip(preds_ols, 0, None)

    # --- 6. PREDIKSI MODEL RANDOM FOREST ---
    df_sim_rf = df_res.copy()
    df_sim_rf[gwr_meta['features_final']] = scaler_rf.transform(df_sim_rf[gwr_meta['features_final']])
    X_rf_final = df_sim_rf[gwr_meta['features_eng']]
    pred_log_rf = rf_model.predict(X_rf_final)
    df_res['Prediksi_Omzet_RF'] = np.expm1(pred_log_rf)

    # --- 7. PREDIKSI MODEL SPASIAL GWR ---
    transformer = Transformer.from_crs("EPSG:4326", "EPSG:32748", always_xy=True)
    preds_gwr = []
    nama_cabang_dekat = []
    jarak_dekat_km = []
    coords_train_utm = df_ref[['x_utm', 'y_utm']].values
    
    for idx, row in df_res.iterrows():
        try:
            x_sim_utm, y_sim_utm = transformer.transform(row['longitude'], row['latitude'])
            
            # Atasi jika koordinat menghasilkan nilai tak terhingga/NaN saat proyeksi
            if np.isnan(x_sim_utm) or np.isinf(x_sim_utm) or np.isnan(y_sim_utm) or np.isinf(y_sim_utm):
                x_sim_utm, y_sim_utm = coords_train_utm[0]  # default ke koordinat referensi pertama
                
            coords_sim_utm = np.array([[x_sim_utm, y_sim_utm]])
            distances = cdist(coords_sim_utm, coords_train_utm)
            nearest_idx = np.argmin(distances[0])
            cabang_terdekat = df_ref.iloc[nearest_idx]
            
            nama_cabang_dekat.append(cabang_terdekat['nama_cabang'])
            jarak_dekat_km.append(distances[0][nearest_idx] / 1000)
            
            X_sim_gwr_raw = row[gwr_meta['features_GW2']].values
            X_sim_gwr_scaled = (X_sim_gwr_raw - gwr_meta['X_gwr_mean']) / gwr_meta['X_gwr_std']
            
            local_betas = [cabang_terdekat['gwr_intercept']] + [cabang_terdekat[f'gwr_beta_{col}'] for col in gwr_meta['features_GW2']]
            X_predict_block = np.hstack([1, X_sim_gwr_scaled])
            pred_std_gwr = np.dot(X_predict_block, local_betas)
            pred_omzet_gwr = (pred_std_gwr * gwr_meta['y_gwr_std']) + gwr_meta['y_gwr_mean']
            
            preds_gwr.append(max(0, pred_omzet_gwr)) # Nilai omzet tidak boleh minus
        except Exception:
            # Fallback aman jika salah satu baris data spasial corrupt
            preds_gwr.append(0.0)
            nama_cabang_dekat.append("Tidak Terdeteksi")
            jarak_dekat_km.append(999.9)

    df_res['Prediksi_Omzet_GWR'] = preds_gwr
    df_res['Cabang_Terdekat_Ref'] = nama_cabang_dekat
    df_res['Jarak_Ref_KM'] = jarak_dekat_km
    
    return df_res

# ========================================================
# 3. NAVIGASI UTAMA (SIDEBAR MENU)
# ========================================================
st.sidebar.title("Navigasi Utama")
menu_terpilih = st.sidebar.radio("Pilih Menu Halaman:", ["Simulasi Cabang Baru", "Performa & Evaluasi Model"])
st.sidebar.markdown("---")
st.sidebar.caption("Data Analytics © 2026")

# ========================================================
# HALAMAN 1: SIMULASI CABANG BARU (BISA BATCH ATAU MANUAL)
# ========================================================
if menu_terpilih == "Simulasi Cabang Baru":
    st.title("Simulasi Perbandingan 3 Model Forecasting Omzet PGI")
    st.write("Sistem ini membandingkan hasil model OLS, Random Forest, dan Koreksi Spasial GWR untuk analisis ekspansi cabang.")

    metode_input = st.radio(
        "Pilih Metode Input Parameter Data:", 
        ["Upload File Batch (Banyak Data)", "Input Manual (Satu-satu seperti dulu)"], 
        horizontal=True
    )

    # --- OPSI 1: UPLOAD BATCH ---
    if metode_input == "Upload File Batch (Banyak Data)":
        st.markdown("### Prediksi Batch Massal via File CSV / Excel")
        st.write("Unggah file hasil data mining Anda di bawah ini:")
        file_diunggah = st.file_uploader("Pilih file CSV atau Excel:", type=['csv', 'xlsx'])
        
        if file_diunggah is not None:
            try:
                if file_diunggah.name.endswith('.csv'):
                    df_batch = pd.read_csv(file_diunggah)
                else:
                    df_batch = pd.read_excel(file_diunggah)
                
                st.success(f"File '{file_diunggah.name}' sukses dimuat! Terbaca {len(df_batch)} baris data.")
                
                tombol_proses = st.button("Jalankan Prediksi Massal", type="primary")
                if tombol_proses:
                    with st.spinner("Sedang memproses kalkulasi seluruh data..."):
                        df_hasil = hitung_prediksi_batch(df_batch, ols_model, scaler_ols, rf_model, scaler_rf, gwr_meta, df_ref)
                        
                    st.markdown("---")
                    st.subheader("Hasil Komparasi Evaluasi Omzet")
                    
                    # Menyusun struktur kolom output tampilan
                    kolom_output = []
                    if 'nama_cabang' in df_hasil.columns:
                        kolom_output.append('nama_cabang')
                    elif 'id_cabang_rencana' in df_hasil.columns:
                        kolom_output.append('id_cabang_rencana')
                    
                    kolom_output.extend(['Prediksi_Omzet_OLS', 'Prediksi_Omzet_RF', 'Prediksi_Omzet_GWR'])
                    
                    # Cek keberadaan nilai aktual
                    if 'avg_omzet' in df_hasil.columns:
                        df_hasil = df_hasil.rename(columns={'avg_omzet': 'Omzet_Actual'})
                        kolom_output.append('Omzet_Actual')
                        has_actual = True
                    else:
                        has_actual = False
                        
                    # Format Rupiah
                    format_dict = {
                        'Prediksi_Omzet_OLS': 'Rp {:,.2f}',
                        'Prediksi_Omzet_RF': 'Rp {:,.2f}',
                        'Prediksi_Omzet_GWR': 'Rp {:,.2f}'
                    }
                    if has_actual:
                        format_dict['Omzet_Actual'] = 'Rp {:,.2f}'
                        
                    st.dataframe(df_hasil[kolom_output].style.format(format_dict), use_container_width=True)
                    
                    st.download_button(
                        label="Unduh Hasil Prediksi Lengkap (.CSV)", 
                        data=df_hasil.to_csv(index=False).encode('utf-8'), 
                        file_name='hasil_prediksi_3model_batch.csv', 
                        mime='text/csv'
                    )
            except Exception as err:
                st.error(f"Gagal memproses file batch. Periksa nama kolom data Anda. Detail Error: {err}")

    # --- OPSI 2: INPUT MANUAL ---
    else:
        st.markdown("### Masukkan Parameter Karakteristik Cabang Secara Manual")
        with st.form("simulation_form"):
            col1, col2 = st.columns(2)
            with col1:
                new_lat = st.number_input("Latitude (Garis Lintang)", value=-6.925914, format="%.6f")
                new_lon = st.number_input("Longitude (Garis Lintang)", value=107.588618, format="%.6f")
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
                df_manual_input = pd.DataFrame([{
                    'latitude': new_lat, 'longitude': new_lon, 'umk': new_umk, 
                    'penduduk': new_penduduk, 'kemiskinan': new_kemiskinan, 
                    'lebar_ruko': new_lebar_ruko, 'jumlah_bangunan': new_jumlah_bangunan, 
                    'jumlah_kompetitor': new_jumlah_kompetitor, 'jumlah_pasar_tradisional': new_jumlah_pasar_tradisional, 
                    'jarak_pasar': new_jarak_pasar, 'jumlah_restoran': new_jumlah_restoran, 
                    'jumlah_fasilitas_belanja': new_jumlah_fasilitas_belanja, 'jumlah_toko_ponsel': new_jumlah_toko_ponsel, 
                    'kategori_wilayah': kategori_wilayah, 'tipe_jalan': tipe_jalan
                }])
                
                df_hasil_manual = hitung_prediksi_batch(df_manual_input, ols_model, scaler_ols, rf_model, scaler_rf, gwr_meta, df_ref)
                row_hasil = df_hasil_manual.iloc[0]
                
                st.markdown("---")
                st.subheader("Hasil Estimasi Komparasi Omzet Bulanan (Manual)")
                m_col1, m_col2, m_col3 = st.columns(3)
                with m_col1:
                    st.metric(label="1. OLS Baseline Model", value=f"Rp {row_hasil['Prediksi_Omzet_OLS']:,.2f}".replace(",", "."))
                    st.caption("Prediksi tren linear global tanpa pembobotan lokal geografis.")
                with m_col2:
                    st.metric(label="2. Random Forest (Optimasi)", value=f"Rp {row_hasil['Prediksi_Omzet_RF']:,.2f}".replace(",", "."))
                    st.caption("Akurat dalam mengenali pola interaksi non-linear parameter bisnis.")
                with m_col3:
                    st.metric(label="3. Spasial GWR Hasil Koreksi", value=f"Rp {row_hasil['Prediksi_Omzet_GWR']:,.2f}".replace(",", "."))
                    st.caption(f"Berbasis bobot lokal titik terdekat: **{row_hasil['Cabang_Terdekat_Ref']}** (Jarak: {row_hasil['Jarak_Ref_KM']:.2f} km).")

# ========================================================
# HALAMAN 2: PERFORMA & EVALUASI MODEL (BAGIAN LAIN KODE TETAP)
# ========================================================
elif menu_terpilih == "Performa & Evaluasi Model":
    st.title("Laporan Metrik Performa & Evaluasi Pemodelan")
    tab_ols, tab_rf, tab_gwr = st.tabs([" 1. Baseline MLR (OLS)", " 2. Random Forest Regressor", " 3. Geographically Weighted Regression (GWR)"])
    
    with tab_ols:
        st.header("Multiple Linear Regression - OLS Baseline")
        c1, c2 = st.columns(2)
        c1.metric(label="R-squared (R²)", value="0.2179")
        c2.metric(label="MAE", value="Rp 160.364.206,25")
        
    with tab_rf:
        st.header("Random Forest Regressor (Optimized)")
        rf1, rf2 = st.columns(2)
        rf1.metric(label="Final Optimized R-squared", value="0.2675")
        rf2.metric(label="Mean Absolute Error (MAE)", value="Rp 154.615.822,10")
        
    with tab_gwr:
        st.header("Geographically Weighted Regression (GWR) Spasial")
        g1, g2, g3 = st.columns(3)
        g1.metric(label="R-squared Global (R²)", value="0.3480")
        g2.metric(label="Adjusted R-squared", value="0.2940")
        g3.metric(label="RMSE Spasial", value="Rp 207.524.678,53")
        
        coef_data = {
            'Nama Parameter/Variabel': [
                'Intercept (Konstanta Spasial)', 'umk', 'penduduk', 'kemiskinan', 
                'jumlah_kompetitor', 'jumlah_pasar_tradisional', 'jarak_pasar', 
                'lebar_ruko', 'jumlah_bangunan', 'kategori_wilayah_mapped', 
                'jalan_mapped', 'commercial_hub_index', 'premium_spot_score', 'comp_per_pop'
            ],
            'Rata-rata Koefisien (Beta)': [
                -0.1700, 0.3753, -0.0221, 0.0065, 0.1421, 0.0439, 0.0633, 
                0.2014, -0.0652, 0.0816, -0.0864, 0.2125, -0.0562, -1.3624
            ]
        }
        df_coef = pd.DataFrame(coef_data)
        st.table(df_coef)
