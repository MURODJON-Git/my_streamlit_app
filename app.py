import streamlit as st
import pandas as pd
import numpy as np

from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA

import plotly.express as px

import umap


st.set_page_config(page_title="Mall Clustering App", layout="wide")

st.title("🛍️ Mall Customers Clustering App")
st.markdown("""Интерактивное приложение для кластеризации клиентов с использованием **K-Means**  и снижения размерности (**PCA / UMAP**). """)

@st.cache_data
def load_data():
    df = pd.read_csv("Mall_Customers.csv")

    df = df.dropna()

    df["Gender"] = df["Gender"].map({"Male": 0, "Female": 1})

    features = ["Age", "Annual Income (k$)", "Spending Score (1-100)", "Gender"]
    X = df[features]

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    return df, X_scaled, features


df, X_scaled, features = load_data()
st.sidebar.header("⚙️ Настройки")

k = st.sidebar.slider("Количество кластеров (K-Means)", 2, 10, 4)

method = st.sidebar.radio(
    "Метод снижения размерности",
    ["PCA", "UMAP"]
)

kmeans = KMeans(n_clusters=k, random_state=42, n_init="auto")
clusters = kmeans.fit_predict(X_scaled)

df["Cluster"] = clusters

if method == "PCA":
    reducer = PCA(n_components=2)
    X_2d = reducer.fit_transform(X_scaled)
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
    title=f"K-Means Clusters + {method}",
    hover_data=["Age", "Annual Income (k$)", "Spending Score (1-100)"]
)

st.plotly_chart(fig, use_container_width=True)


st.subheader("📊 Данные")
st.dataframe(df.head())
