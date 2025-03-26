# Bölüm 1: Giriş, Ayarlar ve Veri Yükleme
import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from collections import Counter
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA

st.set_page_config(page_title="Pokédex Dashboard", layout="wide")

st.title("🧬 Pokémon Veri Analiz Paneli")
st.markdown("Bu panelde Pokémon verileri üzerinde analizler, filtreler ve görselleştirmeler yapabilirsin.")

@st.cache_data
def load_data():
    df = pd.read_csv("pokedex.csv", sep=";")
    df = df[df["type"].apply(lambda x: isinstance(x, str))].copy()
    df["total"] = df[["hp", "attack", "defense", "s_attack", "s_defense", "speed"]].sum(axis=1)
    return df

df = load_data()

# Genel istatistikler
st.subheader("📊 Genel İstatistikler")
col1, col2, col3, col4 = st.columns(4)
col1.metric("Toplam Pokémon", df.shape[0])
col2.metric("Ortalama HP", round(df["hp"].mean(), 1))
col3.metric("Ortalama Attack", round(df["attack"].mean(), 1))
col4.metric("Ortalama Speed", round(df["speed"].mean(), 1))

# Bölüm 2: Tip Dağılımı
st.subheader("🌈 Pokémon Tip Dağılımı")

all_types = df["type"].dropna().astype(str).str.replace(r"[{}]", "", regex=True).str.split(",")
flat_types = [t.strip() for sublist in all_types for t in sublist]
type_counter = Counter(flat_types)

fig1, ax1 = plt.subplots(figsize=(10, 5))
pd.Series(type_counter).sort_values(ascending=False).plot(kind="bar", ax=ax1)
ax1.set_ylabel("Adet")
ax1.set_xlabel("Tür")
ax1.set_title("Pokémon Tip Dağılımı")
st.pyplot(fig1)

# Bölüm 3: Filtreleme ve Arama
st.sidebar.header("🔍 Pokémon Filtrele")
name_search = st.sidebar.text_input("İsme göre ara").lower()
selected_types = st.sidebar.multiselect("Tip Seç", list(set(flat_types)))
min_attack = st.sidebar.slider("Min Attack", 0, 200, 0)
max_attack = st.sidebar.slider("Max Attack", 0, 200, 200)

filtered_df = df.copy()
if name_search:
    filtered_df = filtered_df[filtered_df["name"].str.lower().str.contains(name_search)]
if selected_types:
    filtered_df = filtered_df[filtered_df["type"].apply(lambda x: any(t in x for t in selected_types))]
filtered_df = filtered_df[(filtered_df["attack"] >= min_attack) & (filtered_df["attack"] <= max_attack)]

st.subheader("🔎 Filtrelenmiş Pokémon’lar")
st.dataframe(filtered_df[["name", "type", "hp", "attack", "defense", "speed"]])

# Bölüm 4: En Güçlü 10 Pokémon
st.subheader("💥 En Güçlü 10 Pokémon (Toplam Stat)")
top10 = df.sort_values("total", ascending=False).head(10)
st.dataframe(top10[["name", "total", "hp", "attack", "defense", "speed"]])

# Bölüm 5: Korelasyon Haritası
st.subheader("📈 İstatistik Korelasyonları (Heatmap)")
numeric_cols = ["hp", "attack", "defense", "s_attack", "s_defense", "speed"]
corr = df[numeric_cols].corr()

fig2, ax2 = plt.subplots()
sns.heatmap(corr, annot=True, cmap="coolwarm", ax=ax2)
st.pyplot(fig2)

# Bölüm 6: Stat Dağılımı (Scatterplot)
st.subheader("📉 Attack vs. Defense Dağılımı")
fig3, ax3 = plt.subplots()
ax3.scatter(df["attack"], df["defense"], alpha=0.6)
ax3.set_xlabel("Attack")
ax3.set_ylabel("Defense")
ax3.set_title("Attack vs. Defense")
st.pyplot(fig3)

# Bölüm 7: Benzer Pokémon Bul (Stat bazlı)
st.subheader("🧠 Statlara Göre Benzer Pokémon’ları Bul")

selected_pokemon = st.selectbox("Bir Pokémon seç", df["name"].values)
selected_row = df[df["name"] == selected_pokemon]
features = df[numeric_cols]
scaler = StandardScaler()
features_scaled = scaler.fit_transform(features)

cos_sim = cosine_similarity(features_scaled)
df["similarity"] = cos_sim[df[df["name"] == selected_pokemon].index[0]]
similar_pokemon = df.sort_values("similarity", ascending=False).iloc[1:6]

st.write(f"**{selected_pokemon}** Pokémon'una stat olarak en çok benzeyenler:")
st.dataframe(similar_pokemon[["name", "similarity", "hp", "attack", "defense", "speed"]])

# Bölüm 8: Statlara Göre Pokémon Kümeleri (Clustering)
st.subheader("🧪 Pokémon Stat Kümeleri (K-Means Clustering)")

num_clusters = st.slider("Kaç küme olsun?", 2, 10, 4)
kmeans = KMeans(n_clusters=num_clusters, random_state=42)
cluster_features = df[numeric_cols]
scaled_features = StandardScaler().fit_transform(cluster_features)
df["cluster"] = kmeans.fit_predict(scaled_features)

# PCA ile 2D görselleştirme
pca = PCA(n_components=2)
pca_result = pca.fit_transform(scaled_features)
df["pca1"] = pca_result[:, 0]
df["pca2"] = pca_result[:, 1]

fig4, ax4 = plt.subplots()
scatter = ax4.scatter(df["pca1"], df["pca2"], c=df["cluster"], cmap="tab10", alpha=0.7)
ax4.set_title("Pokémon Statlarına Göre Küme Dağılımı (PCA)")
ax4.set_xlabel("PCA 1")
ax4.set_ylabel("PCA 2")
st.pyplot(fig4)

# Bölüm 9: Boy vs Kilo Dağılımı
st.subheader("📏 Boy ve Kilo Dağılımı")

fig5, ax5 = plt.subplots()
ax5.scatter(df["height"], df["weight"], alpha=0.6)
ax5.set_xlabel("Boy (dm)")
ax5.set_ylabel("Kilo (hg)")
ax5.set_title("Boy vs Kilo")
st.pyplot(fig5)

