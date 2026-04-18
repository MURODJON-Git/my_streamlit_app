import streamlit as st 
import pandas as pd
import numpy as np

from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.metrics import silhouette_score

import plotly.express as px
import umap
import seaborn as sns
import matplotlib.pyplot as plt

st.set_page_config(page_title="Mall Clustering App", layout="wide")

st.title("🛍️ Mall Customers Clustering App")
st.markdown("Интерактивное приложение для кластеризации клиентов")

@st.cache_data
def load_data():
    df = pd.read_csv("Mall_Customers.csv")
    df = df.dropna()
    df["Gender"] = df["Gender"].map({"Male": 0, "Female": 1})
    return df

df = load_data()

st.sidebar.header("⚙️ Настройки")

features_all = ["Age", "Annual Income (k$)", "Spending Score (1-100)", "Gender"]

selected_features = st.sidebar.multiselect(
    "Выбери признаки",
    features_all,
    default=features_all
)

auto_k = st.sidebar.checkbox("🔍 Авто-выбор лучшего K")
k = st.sidebar.slider("Количество кластеров (K-Means)", 2, 10, 4)

method = st.sidebar.radio(
    "Метод снижения размерности",
    ["PCA", "UMAP"]
)

X = df[selected_features]

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

if auto_k:
    best_k = 2
    best_score = -1

    for i in range(2, 11):
        km = KMeans(n_clusters=i, random_state=42, n_init="auto")
        labels = km.fit_predict(X_scaled)
        score = silhouette_score(X_scaled, labels)

        if score > best_score:
            best_score = score
            best_k = i

    k = best_k
    st.sidebar.success(f"🔥 Лучший K найден: {k}")

kmeans = KMeans(n_clusters=k, random_state=42, n_init="auto")
clusters = kmeans.fit_predict(X_scaled)

df["Cluster"] = clusters

sil_score = silhouette_score(X_scaled, clusters)
st.sidebar.metric("Silhouette Score", f"{sil_score:.3f}")

if method == "PCA":
    reducer = PCA(n_components=2)
else:
    reducer = umap.UMAP(n_components=2, random_state=42)

X_2d = reducer.fit_transform(X_scaled)

df["Dim1"] = X_2d[:, 0]
df["Dim2"] = X_2d[:, 1]

fig = px.scatter(
    df,
    x="Dim1",
    y="Dim2",
    color=df["Cluster"].astype(str),
    title=f"K-Means + {method}",
    hover_data=selected_features
)

st.plotly_chart(fig, use_container_width=True)

with st.expander("📊 Корреляция признаков"):
    corr = df[selected_features].corr()

    fig_corr, ax = plt.subplots()
    sns.heatmap(
        corr,
        annot=True,
        fmt=".2f",
        cmap="coolwarm",
        linewidths=0.5,
        ax=ax
    )

    st.pyplot(fig_corr)

if len(selected_features) >= 3:
    st.subheader("🌐 3D визуализация")
    pca_3d = PCA(n_components=3)
    X_3d = pca_3d.fit_transform(X_scaled)

    df["PC1"] = X_3d[:, 0]
    df["PC2"] = X_3d[:, 1]
    df["PC3"] = X_3d[:, 2]

    fig3d = px.scatter_3d(
        df,
        x="PC1",
        y="PC2",
        z="PC3",
        color=df["Cluster"].astype(str),
        opacity=0.8,
        size=df["Spending Score (1-100)"],
        hover_data=selected_features,
        title="3D Clusters"
    )

    fig3d.update_layout(
        margin=dict(l=0, r=0, b=0, t=40)
    )

    st.plotly_chart(fig3d, use_container_width=True)

else:
    st.warning("⚠️ Для 3D нужно минимум 3 признака")

with st.expander("📊 Показать данные"):
    st.dataframe(df)

with st.expander("📈 Silhouette Score vs K"):
    scores = []
    k_range = range(2, 11)

    for i in k_range:
        km = KMeans(n_clusters=i, random_state=42, n_init="auto")
        labels = km.fit_predict(X_scaled)
        score = silhouette_score(X_scaled, labels)
        scores.append(score)

    fig2 = px.line(
        x=list(k_range),
        y=scores,
        markers=True,
        title="Silhouette Score vs K"
    )

    st.plotly_chart(fig2)

csv = df.to_csv(index=False).encode('utf-8')

st.download_button(
    label="📥 Скачать результат",
    data=csv,
    file_name="clustered_data.csv",
    mime="text/csv"
)
