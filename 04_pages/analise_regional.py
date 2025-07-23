import streamlit as st
import pandas as pd

@st.cache_data
def carregar_dados():
    base = "https://github.com/Kinrider/tech_challenge_5/raw/main"
    df_candidatos = pd.read_parquet(f"{base}/01_fontes/arquivos_decision/fontes_tratadas/01_candidatos.parquet", engine="pyarrow")
    df_clusters = pd.read_parquet(f"{base}/03_modelos/04_clusterizados.parquet", engine="pyarrow")
    return df_candidatos, df_clusters

def render():
    st.title("🌍 Análise Regional do Candidato")
    candidato = st.session_state.get("candidato_classificado")
    if not candidato:
        st.warning("⚠️ Classifique um candidato primeiro na aba anterior.")
        return

    df_candidatos, df_clusters = carregar_dados()
    cluster = candidato["cluster"]
    municipio = candidato["municipio"]

    df_filtrado = df_candidatos[
        (df_clusters["cluster"] == cluster) &
        (df_candidatos["municipio"].str.lower() == municipio.lower())
    ]

    st.subheader(f"📍 Perfil dos candidatos em **{municipio}** do **Cluster {cluster}**")
    if df_filtrado.empty:
        st.error("Nenhum outro candidato encontrado com as mesmas condições.")
    else:
        st.metric("Quantidade de candidatos na região", len(df_filtrado))
        st.write("### Distribuição por escolaridade:")
        st.write(df_filtrado["nivel_educacional"].value_counts())
        st.write("### Faixa etária média:")
        st.write(round(df_filtrado["idade"].mean(), 1))
        st.write("### Presença de experiência com SAP:")
        st.write(df_filtrado["experiencia_sap"].value_counts(normalize=True))
