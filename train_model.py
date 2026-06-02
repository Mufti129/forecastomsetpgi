import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestRegressor
import statsmodels.api as sm
import joblib

# 1. Load Data
df = pd.read_excel('FIX_mining_prediksi_attribute_jumlah.xlsx')
df_clean = df.copy()

# 2. Mapping Kategorik & Pembersihan
kategori_wilayah_mapping = {'Perdesaan': 0, 'Perkampungan': 1, 'Perkotaan': 2}
df_clean['kategori_wilayah_mapped'] = df_clean['kategori_wilayah'].map(kategori_wilayah_mapping)

jalan_mapping = {'primary': 0, 'residential': 1, 'tertiary': 2, 'secondary': 3, 'living_street': 4, 'trunk': 5}
df_clean['jalan_mapped'] = df_clean['jalan'].map(jalan_mapping)

numeric_cols = df_clean.select_dtypes(include=[np.number]).columns
df_clean[numeric_cols] = df_clean[numeric_cols].fillna(df_clean[numeric_cols].median())

# 3. Feature Engineering
df_clean['commercial_hub_index'] = df_clean['jumlah_restoran'] + df_clean['jumlah_fasilitas_belanja'] + df_clean['jumlah_toko_ponsel']
df_clean['premium_spot_score'] = df_clean['lebar_ruko'] * df_clean['umk']
df_clean['comp_per_pop'] = df_clean['jumlah_kompetitor'] / (df_clean['penduduk'] + 1)

# Fitur yang digunakan (disesuaikan dengan hasil seleksi fitur/VIF skrip asli)
features_eng = [
    'umk','penduduk','kemiskinan','jumlah_fasilitas_belanja','jumlah_toko_ponsel',
    'jumlah_kompetitor','jumlah_pasar_tradisional','jarak_pasar','jumlah_restoran',
    'lebar_ruko','jalan_mapped','jumlah_bangunan','commercial_hub_index',
    'premium_spot_score','comp_per_pop'
]

X = df_clean[features_eng]
y_log = np.log1p(df_clean['avg_omzet'])

# 4. Scaling
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# --- A. Train Model 1: Random Forest ---
rf_model = RandomForestRegressor(n_estimators=500, max_depth=12, min_samples_split=10, random_state=42)
rf_model.fit(X_scaled, y_log)

# --- B. Train Model 2: OLS (Linear Regression) ---
X_scaled_dns = sm.add_constant(X_scaled) # Tambah konstanta untuk OLS
ols_model = sm.OLS(y_log, X_scaled_dns).fit()

# --- C. Persiapan Model 3: GWR (Geographically Weighted Regression) ---
# Mengonversi koordinat ke UTM Zone 48S untuk perhitungan jarak meteran
from pyproj import Transformer
transformer = Transformer.from_crs("EPSG:4326", "EPSG:32748", always_xy=True)
df_clean['x_utm'], df_clean['y_utm'] = transformer.transform(df_clean['longitude'].values, df_clean['latitude'].values)

# Di skrip asli, GWR menghasilkan koefisien lokal (Beta) untuk SETIAP cabang historis.
# Kita akan simpan koefisien GWR ini langsung di dalam dataframe referensi spasial.
from mgwr.gwr import GWR
from mgwr.sel_bw import Sel_BW

coords = df_clean[['x_utm', 'y_utm']].values
y_gwr = df_clean['avg_omzet'].values.reshape(-1, 1)
X_gwr = df_clean[features_eng].values

# Mencari bandwidth optimal untuk GWR
bw_selected = Sel_BW(coords, y_gwr, X_gwr, fixed=False, kernel='bisquare').select()
gwr_model = GWR(coords, y_gwr, X_gwr, bw_selected, fixed=False, kernel='bisquare').fit()

# Simpan intersep dan koefisien lokal GWR ke dataframe agar bisa dipanggil secara instan di web
df_clean['gwr_intercept'] = gwr_model.params[:, 0]
for idx, col_name in enumerate(features_eng):
    df_clean[f'gwr_beta_{col_name}'] = gwr_model.params[:, idx + 1]

# 5. Ekspor semua objek ke file pkl
joblib.dump(rf_model, 'rf_model.pkl')
joblib.dump(ols_model, 'ols_model.pkl')
joblib.dump(scaler, 'scaler.pkl')
df_clean.to_pickle('df_spatial_reference.pkl')

print("Semua model (OLS, RF, GWR-Spasial) berhasil dilatih dan disimpan!")
