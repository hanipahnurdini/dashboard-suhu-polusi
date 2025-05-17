import streamlit as st
import pandas as pd
import plotly.express as px 
import os
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler

BASE_DIR = os.path.dirname(__file__)
data_path = os.path.join(BASE_DIR, "main_data.csv")
df_air_quality = pd.read_csv(data_path)

df_air_quality["year"] = pd.to_datetime(df_air_quality["year"], format='%Y').dt.year  # Convert to int
df_air_quality.fillna(df_air_quality.select_dtypes(include=['number']).mean(numeric_only=True), inplace=True)

av_path = os.path.join(BASE_DIR, "av_suhu_kota_pertahun.csv")
av_suhu_kota_pertahun = pd.read_csv(av_path)

st.set_page_config(
    page_title="Dashboard Suhu & Polutan Kota",
    layout="wide",
    initial_sidebar_state="expanded"
)

st.sidebar.image("https://upload.wikimedia.org/wikipedia/commons/thumb/f/fa/Flag_of_the_People%27s_Republic_of_China.svg/200px-Flag_of_the_People%27s_Republic_of_China.svg.png",
                 use_column_width=True)

st.sidebar.title("\U0001F3D9️ Navigasi")
page = st.sidebar.radio("Pilih Halaman", ["\U0001F4CA Data", "\U0001F4C8 Visualisasi", "\U0001F52C Analisis Lanjutan"])

st.sidebar.subheader("Filter Tahun")
selected_year = st.sidebar.selectbox("Pilih Tahun", sorted(df_air_quality["year"].unique()))
df_filtered = df_air_quality[df_air_quality["year"] == selected_year]

city_stats = df_filtered.groupby("City")[["TEMP", "PM2.5"]].mean().reset_index()
city_stats = city_stats.sort_values(by="TEMP", ascending=False)

st.title("\U0001F30D Analisis Suhu dan Kualitas Udara di Kota-Kota China Tahun 2013–2017")
st.markdown("---")

if not city_stats.empty:
    hottest_city = city_stats.iloc[0]
    coldest_city = city_stats.iloc[-1]

    col1, col2 = st.columns(2)
    with col1:
        st.metric(label="\U0001F321️ Kota Terpanas", 
                  value=hottest_city["City"], 
                  delta=f"{hottest_city['TEMP']:.2f}°C")
    with col2:
        st.metric(label="\u2744️ Kota Terdingin", 
                  value=coldest_city["City"], 
                  delta=f"{coldest_city['TEMP']:.2f}°C")
    
    st.markdown("---")

 # ---- PAGE: Data ----
if page == "📊 Data":
    st.subheader("📊 Data Suhu dan Polusi Udara Tiap Kota")
    all_pollutants = ["TEMP", "PM2.5", "PM10", "SO2", "NO2", "CO", "O3"]
    city_data = df_filtered.groupby("City")[all_pollutants].mean().reset_index()
    st.dataframe(city_data.style.format("{:.2f}"))

    st.markdown("---")
    st.subheader("📅 Rata-rata Suhu per Kota per Tahun")
    av_filtered = av_suhu_kota_pertahun[av_suhu_kota_pertahun["year"] == selected_year]
    st.dataframe(av_filtered.style.format({"TEMP": "{:.2f}"}))

# ---- PAGE: Visualisasi ----
elif page == "📈 Visualisasi":
    st.header("📈 Rata-rata Suhu per Kota (Tahun {})".format(selected_year))
    avg_temp = df_filtered.groupby("City")["TEMP"].mean().sort_values(ascending=False)
    st.bar_chart(avg_temp)

    st.subheader("📊 Heatmap Rata-rata Polutan")
    avg_pollutants = df_filtered.groupby("City")[pollutants].mean()
    fig, ax = plt.subplots(figsize=(12, 6))
    sns.heatmap(avg_pollutants, cmap="coolwarm", annot=True, fmt=".1f", ax=ax)
    st.pyplot(fig)

    st.subheader("🕐 Polutan Berdasarkan Jam (Rata-rata Tiap Kota)")
    fig2, axes = plt.subplots(3, 5, figsize=(20, 10), sharex=True)
    for i, city in enumerate(df["City"].unique()):
        ax = axes.flatten()[i]
        hourly = df[df["City"] == city].groupby("hour")[pollutants].mean()
        hourly.plot(ax=ax)
        ax.set_title(city)
        ax.set_xlabel("Jam")
        ax.set_ylabel("Rata-rata")
        ax.legend().set_visible(False)
    fig2.tight_layout()
    st.pyplot(fig2)

# ---- PAGE: Analisis Lanjutan ----
elif page == "🧪 Analisis Lanjutan":
    st.header("🧪 Clustering Kota Berdasarkan Polutan dan Suhu")
    cluster_df = df.groupby("City")[pollutants + ["TEMP"]].mean().dropna()

    scaler = StandardScaler()
    scaled = scaler.fit_transform(cluster_df)

    inertia = []
    for k in range(1, 10):
        kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
        kmeans.fit(scaled)
        inertia.append(kmeans.inertia_)

    st.subheader("Elbow Method untuk Menentukan k")
    fig_elbow, ax_elbow = plt.subplots()
    ax_elbow.plot(range(1, 10), inertia, marker='o')
    ax_elbow.set_xlabel("Jumlah Cluster")
    ax_elbow.set_ylabel("Inertia")
    ax_elbow.set_title("Elbow Method")
    st.pyplot(fig_elbow)

    optimal_k = 3
    model = KMeans(n_clusters=optimal_k, random_state=42, n_init=10)
    cluster_df["Cluster"] = model.fit_predict(scaled)

    st.subheader("Visualisasi Clustering (PM2.5 vs PM10)")
    fig_scatter, ax = plt.subplots()
    sns.scatterplot(data=cluster_df, x="PM2.5", y="PM10", hue="Cluster", palette="Set2", ax=ax)
    ax.set_title("Clustering Kota Berdasarkan PM2.5 dan PM10")
    st.pyplot(fig_scatter)

    st.dataframe(cluster_df.reset_index())
