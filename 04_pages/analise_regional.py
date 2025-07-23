import streamlit as st
import pandas as pd
import plotly.express as px

@st.cache_data
def carregar_base_sumarizada():
    url = "https://github.com/Kinrider/tech_challenge_5/raw/main/01_fontes/arquivos_decision/fontes_tratadas/05_dados_sumarizados.xlsx"
    return pd.read_excel(url)

def render():
    st.title("🌍 Análise Regional do Candidato")

    candidato = st.session_state.get("candidato_classificado")
    if not candidato:
        st.warning("⚠️ Classifique um candidato primeiro na aba 'Novo Candidato'.")
        return

    estado = candidato["estado"]
    cluster = candidato["cluster"]
    nome = candidato["nome"]

    df = carregar_base_sumarizada()

    # Filtrar dados para estado + cluster
    df_filtro = df[(df["estado"] == estado) & (df["cluster"] == cluster)]

    if df_filtro.empty:
        st.error("❌ Nenhum dado encontrado para o cluster e estado informados.")
        return

    st.subheader(f"📍 Perfil no Estado **{estado}** — Cluster {cluster}")
    st.write(f"👤 Candidato: **{nome}**")

    st.metric("Total de candidatos", int(df_filtro["candidatos"].sum()))

    st.write("### Distribuição por faixa etária")
    fig = px.bar(
        df_filtro,
        x="faixa_etaria",
        y="candidatos",
        labels={"faixa_etaria": "Faixa Etária", "candidatos": "Qtd. Candidatos"},
        title="📊 Quantidade de candidatos por faixa etária",
        color_discrete_sequence=["#636EFA"]
    )
    st.plotly_chart(fig, use_container_width=True)

    st.write("### Outros indicadores médios por faixa etária")
    st.dataframe(
        df_filtro[[
            "faixa_etaria",
            "escolaridade_media",
            "perc_com_sap",
            "remuneracao_media",
            "experiencias_media"
        ]].set_index("faixa_etaria").round(2)
    )
