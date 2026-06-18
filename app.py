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
# 2. FUNGSIONALITAS UTAMA PREDIKSI BATCH
# ========================================================
def hitung_prediksi_batch(df_input, ols_model, scaler_ols, rf_model, scaler_rf, gwr_meta, df_ref):
    df_res = df_input.copy()
    
    # Penyelarasan nama kolom jika pengguna menggunakan singkatan (lat/lon)
    if 'lat' in df_res.columns and 'latitude' not in df_res.columns:
        df_res['latitude'] = df_res['lat']
    if 'lon' in df_res.columns and 'longitude' not in df_res.columns:
        df_res['longitude'] = df_res['lon']
    if 'jalan' in df_res.columns and 'tipe_jalan' not in df_res.columns:
        df_res['tipe_jalan'] = df_res['jalan']

    # Mapping Kategori & Jalan ke bentuk numerik
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
    
    preds_ols = ols_model.predict(X_ols_const)
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
    
    coords_train_utm = df_ref[['x_utm', 'y_utm']].values

    for idx, row in df_res.iterrows():
        x_sim_utm, y_sim_utm = transformer.transform(row['longitude'], row['latitude'])
        coords_sim_utm = np.array([[x_sim_utm, y_sim_utm]])
        
        distances = cdist(coords_sim_utm, coords_train_utm)
        nearest_idx = np.argmin(distances[0])
        cabang_terdekat = df_ref.iloc[nearest_idx]

        X_sim_gwr_raw = row[gwr_meta['features_GW2']].values
        X_sim_gwr_scaled = (X_sim_gwr_raw - gwr_meta['X_gwr_mean']) / gwr_meta['X_gwr_std']
        
        local_betas = [cabang_terdekat['gwr_intercept']] + [cabang_terdekat[f'gwr_beta_{col}'] for col in gwr_meta['features_GW2']]
        X_predict_block = np.hstack([1, X_sim_gwr_scaled])
        
        pred_std_gwr = np.dot(X_predict_block, local_betas)
        pred_omzet_gwr = (pred_std_gwr * gwr_meta['y_gwr_std']) + gwr_meta['y_gwr_mean']
        preds_gwr.append(pred_omzet_gwr)

    df_res['Prediksi_Omzet_GWR'] = preds_gwr
    
    return df_res


# ========================================================
# 3. NAVIGASI UTAMA (SIDEBAR MENU)
# ========================================================
st.sidebar.title("Navigasi Aplikasi")
menu_terpilih = st.sidebar.radio(
    "Pilih Menu Halaman:",
    ["Simulasi & Evaluasi Batch", "Performa Global Model"]
)
st.sidebar.markdown("---")
st.sidebar.caption("Data Analytics © 2026")


# ========================================================
# HALAMAN 1: SIMULASI & EVALUASI BATCH (TARGET UTAMA)
# ========================================================
if menu_terpilih == "Simulasi & Evaluasi Batch":
    st.title("📊 Validasi & Prediksi Batch 3 Model vs Omzet Actual")
    st.write("Halaman ini memungkinkan Anda mengunggah file data atribut retail untuk memproses prediksi secara massal sekaligus membandingkannya dengan target omzet aktual.")

    file_diunggah = st.file_uploader("Upload file CSV atau Excel Hasil Mining Anda:", type=['csv', 'xlsx'])
    
    if file_diunggah is not None:
        try:
            if file_diunggah.name.endswith('.csv'):
                df_batch = pd.read_csv(file_diunggah)
            else:
                df_batch = pd.read_excel(file_diunggah)
            
            st.success(f"File '{file_diunggah.name}' berhasil dimuat! Mendeteksi {len(df_batch)} baris data.")
            
            tombol_proses = st.button("Jalankan Komparasi Prediksi Massal", type="primary")
            
            if tombol_proses:
                with st.spinner("Menghitung kalkulasi OLS, Random Forest, dan Spasial GWR..."):
                    df_hasil = hitung_prediksi_batch(df_batch, ols_model, scaler_ols, rf_model, scaler_rf, gwr_meta, df_ref)
                
                st.markdown("---")
                st.subheader("📋 Tabel Hasil Evaluasi Omzet Cabang")
                
                # Menyusun kolom tampilan sesuai request pengguna
                kolom_output = []
                if 'nama_cabang' in df_hasil.columns:
                    kolom_output.append('nama_cabang')
                elif 'id_cabang_rencana' in df_hasil.columns:
                    kolom_output.append('id_cabang_rencana')
                
                kolom_output.extend(['Prediksi_Omzet_OLS', 'Prediksi_Omzet_RF', 'Prediksi_Omzet_GWR'])
                
                # Cek jika ada data actual/asli dari file mining Anda
                if 'avg_omzet' in df_hasil.columns:
                    kolom_output.append('avg_omzet')
                    df_hasil = df_hasil.rename(columns={'avg_omzet': 'Omzet_Actual'})
                    index_actual = 'Omzet_Actual'
                else:
                    index_actual = None
                
                # Format rupiah untuk visualisasi dataframe
                format_dict = {
                    'Prediksi_Omzet_OLS': 'Rp {:,.2f}',
                    'Prediksi_Omzet_RF': 'Rp {:,.2f}',
                    'Prediksi_Omzet_GWR': 'Rp {:,.2f}'
                }
                if index_actual:
                    format_dict['Omzet_Actual'] = 'Rp {:,.2f}'
                
                # Menampilkan tabel interaktif di Streamlit
                st.dataframe(
                    df_hasil[kolom_output].style.format(format_dict), 
                    use_container_width=True
                )
                
                # Tombol Download untuk menyimpan hasil lengkap ke komputer Anda
                st.download_button(
                    label="📥 Download Hasil Prediksi Lengkap (.CSV)",
                    data=df_hasil.to_csv(index=False).encode('utf-8'),
                    file_name='hasil_komparasi_omzet_3model.csv',
                    mime='text/csv'
                )
        except Exception as err:
            st.error(f"Gagal memproses file. Pastikan format kolom sesuai dengan data mining. Detail Error: {err}")


# ========================================================
# HALAMAN 2: PERFORMA GLOBAL MODEL
# ========================================================
elif menu_terpilih == "Performa Global Model":
    st.title("Laporan Metrik Performa & Evaluasi Pemodelan")
    
    tab_ols, tab_rf, tab_gwr = st.tabs(["1. Baseline OLS", "2. Random Forest", "3. Spasial GWR"])
    
    with tab_ols:
        st.header("Multiple Linear Regression - OLS Baseline")
        c1, c2 = st.columns(2)
        c1.metric(label="R-squared (R²)", value="0.2179")
        c2.metric(label="MAE", value="Rp 160.364.206,25")

    with tab_rf:
        st.header("Random Forest Regressor")
        rf1, rf2 = st.columns(2)
        rf1.metric(label="Optimized R-squared", value="0.2675")
        rf2.metric(label="Mean Absolute Error (MAE)", value="Rp 154.615.822,10")

    with tab_gwr:
        st.header("Geographically Weighted Regression (GWR)")
        g1, g2, g3 = st.columns(3)
        g1.metric(label="R-squared Global (R²)", value="0.3480")
        g2.metric(label="Adjusted R-squared", value="0.2940")
        g3.metric(label="RMSE Spasial", value="Rp 207.524.678,53")
