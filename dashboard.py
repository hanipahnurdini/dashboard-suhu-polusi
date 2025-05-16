import streamlit as st
import pandas as pd
import plotly.express as px
import os
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler

# Load dataset
BASE_DIR = os.path.dirname(__file__)
data_path = os.path.join(BASE_DIR, "main_data.csv")
df_air_quality = pd.read_csv(data_path)

# Preprocessing
df_air_quality["year"] = pd.to_datetime(df_air_quality["year"], format='%Y').dt.year

# Isi missing values dengan rata-rata numerik
df_air_quality.fillna(df_air_quality.select_dtypes(include=['number']).mean(numeric_only=True), inplace=True)

# Konfigurasi halaman Streamlit
st.set_page_config(
    page_title="Dashboard Analisis Kualitas Udara dan Suhu Kota di China",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Sidebar Navigasi
st.sidebar.image("https://upload.wikimedia.org/wikipedia/commons/thumb/f/fa/Flag_of_the_People%27s_Republic_of_China.svg/200px-Flag_of_the_People%27s_Republic_of_China.svg.png",
                 use_column_width=True)
st.sidebar.title("\U0001F3D9️ Navigasi")
page = st.sidebar.radio("Pilih Halaman", ["\U0001F4CA Data", "\U0001F4C8 Visualisasi", "\U0001F52C Analisis Lanjutan"])

# Sidebar filter tahun
st.sidebar.subheader("Filter Tahun")
selected_year = st.sidebar.selectbox("Pilih Tahun", sorted(df_air_quality["year"].unique()))
df_filtered = df_air_quality[df_air_quality["year"] == selected_year]

# Statistik kota berdasarkan tahun
city_stats = df_filtered.groupby("City")["TEMP"].mean().reset_index()
city_stats = city_stats.sort_values(by="TEMP", ascending=False)

st.title("\U0001F30D Dashboard Suhu dan Kualitas Udara Kota di China (2013-2017)")
st.markdown("---")

if not city_stats.empty:
    hottest_city = city_stats.iloc[0]
    coldest_city = city_stats.iloc[-1]

    col1, col2 = st.columns(2)
    with col1:
        st.metric(label="\U0001F321️ Kota Terpanas", value=hottest_city["City"], delta=f"{hottest_city['TEMP']:.2f}°C")
    with col2:
        st.metric(label="\u2744️ Kota Terdingin", value=coldest_city["City"], delta=f"{coldest_city['TEMP']:.2f}°C")

    st.markdown("---")

    if page == "\U0001F4CA Data":
        st.subheader("\U0001F4CA Data Suhu dan Polusi Udara Tiap Kota")
        all_pollutants = ["TEMP", "PM2.5", "PM10", "SO2", "NO2", "CO", "O3"]
        city_data = df_filtered.groupby("City")[all_pollutants].mean().reset_index()
        st.dataframe(city_data.style.format("{:.2f}"))

    elif page == "\U0001F4C8 Visualisasi":
        st.subheader("\U0001F321️ Suhu Rata-rata Kota Tahun Terpilih")
        fig_temp = px.bar(city_stats, x="TEMP", y="City", orientation="h",
                          title=f"Suhu Rata-rata per Kota ({selected_year})",
                          labels={"TEMP": "Suhu (°C)", "City": "Kota"},
                          color="TEMP", color_continuous_scale="YlOrRd")
        st.plotly_chart(fig_temp, use_container_width=True)

        st.subheader("\U0001F4A8 Kota dengan Polusi Udara Tertinggi Tiap Tahun")
        city_max_pollution = df_air_quality.loc[df_air_quality.groupby("year")["PM2.5"].idxmax()]
        fig_pollution = px.bar(city_max_pollution, x="year", y="PM2.5", color="City",
                               title="Konsentrasi PM2.5 Tertinggi per Tahun",
                               labels={"PM2.5": "Konsentrasi (µg/m³)"})
        st.plotly_chart(fig_pollution, use_container_width=True)

        st.subheader("\U0001F30E Urtan Kota Paling Panas dan Dingin Sepanjang 2013–2017")
        df_all_years = df_air_quality.groupby("City")["TEMP"].mean().reset_index()
        col1, col2 = st.columns(2)
        with col1:
            fig_hot = px.bar(df_all_years.nlargest(12, "TEMP"), x="TEMP", y="City", orientation="h",
                             title="Urutan Kota Terpanas", color="TEMP",
                             color_continuous_scale="Reds")
            st.plotly_chart(fig_hot, use_container_width=True)

        with col2:
            fig_cold = px.bar(df_all_years.nsmallest(12, "TEMP"), x="TEMP", y="City", orientation="h",
                              title="Urutan Kota Terdingin", color="TEMP",
                              color_continuous_scale="Blues")
            st.plotly_chart(fig_cold, use_container_width=True)

    elif page == "\U0001F52C Analisis Lanjutan":
        st.title("\U0001F52C Analisis Clustering Berdasarkan Suhu dan Polutan")

        features = ["TEMP", "PM2.5", "PM10", "SO2", "NO2", "CO", "O3"]
        df_cluster = df_air_quality[features].dropna()

        scaler = StandardScaler()
        df_scaled = scaler.fit_transform(df_cluster)

        inertia = []
        for k in range(1, 10):
            kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
            kmeans.fit(df_scaled)
            inertia.append(kmeans.inertia_)

        st.subheader("\U0001F52C Menentukan Jumlah Cluster - Elbow Method")
        fig, ax = plt.subplots()
        ax.plot(range(1, 10), inertia, marker='o')
        ax.set_xlabel('Jumlah Cluster')
        ax.set_ylabel('Inertia')
        ax.set_title('Elbow Method untuk Menentukan K')
        st.pyplot(fig)

        st.subheader("\U0001F5FA️ Visualisasi Clustering (TEMP vs Polutan)")
        k_optimal = 3
        kmeans = KMeans(n_clusters=k_optimal, random_state=42, n_init=10)
        df_air_quality['Cluster'] = kmeans.fit_predict(df_scaled)

        fig2, ax2 = plt.subplots()
        sns.scatterplot(data=df_air_quality, x="TEMP", y="PM2.5", hue="Cluster", palette="viridis", ax=ax2)
        ax2.set_title("Clustering Berdasarkan Suhu dan Polutan")
        st.pyplot(fig2)

        st.markdown("---")
        st.subheader("\U0001F4D1 Fitur yang Digunakan dalam Clustering")
        st.markdown(
            "- TEMP (Suhu)")
        st.markdown("- PM2.5")
        st.markdown("- PM10")
        st.markdown("- SO2")
        st.markdown("- NO2")
        st.markdown("- CO")
        st.markdown("- O3")
