import pandas as pd
import joblib
from sklearn.cluster import KMeans

# === Caminho do dataset de entrada ===
input = 'https://github.com/Kinrider/tech_challenge_5/raw/refs/heads/main/01_fontes/arquivos_decision/fontes_tratadas/02_input_Kmeans.parquet'
df = pd.read_parquet(input)

# === Remover colunas não utilizadas ===
if "cluster" in df.columns:
    df = df.drop(columns=["cluster"])

# === Treinar modelo KMeans ===
kmeans = KMeans(n_clusters=4, random_state=42, n_init=10)
kmeans.fit(df)

# === Salvar modelo e colunas usadas ===
joblib.dump(kmeans, "03_modelos/modelo_kmeans.joblib")
joblib.dump(df.columns.tolist(), "03_modelos/colunas_usadas.joblib")

print("✅ Modelo e colunas salvos em: 03_modelos/")
