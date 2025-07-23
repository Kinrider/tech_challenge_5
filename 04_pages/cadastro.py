import streamlit as st
import pandas as pd
import joblib
import requests
from io import BytesIO

@st.cache_resource
def carregar_recursos():
    BASE_GITHUB = "https://github.com/Kinrider/tech_challenge_5/raw/main"
    modelo = joblib.load(BytesIO(requests.get(f"{BASE_GITHUB}/03_modelos/modelo_kmeans.joblib").content))
    colunas_modelo = joblib.load(BytesIO(requests.get(f"{BASE_GITHUB}/03_modelos/colunas_usadas.joblib").content))
    return modelo, colunas_modelo

@st.cache_data
def carregar_parametros_zscore():
    BASE_GITHUB = "https://github.com/Kinrider/tech_challenge_5/raw/main"
    df = pd.read_parquet('https://github.com/Kinrider/tech_challenge_5/raw/refs/heads/main/01_fontes/arquivos_decision/fontes_tratadas/remuneracao_mensal_only.parquet', engine="pyarrow")
    media = df["remuneracao_mensal_brl"].replace(-9999, pd.NA).dropna().mean()
    desvio = df["remuneracao_mensal_brl"].replace(-9999, pd.NA).dropna().std()
    return media, desvio

def preparar_input(form_dict, colunas_modelo):
    df_input = pd.DataFrame([form_dict])
    
    mapa_educacional = {
        'ensino fundamental': 1,
        'ensino médio': 2,
        'ensino superior': 2,
        'superior incompleto': 2,
        'pós-graduação ou mais': 3,
        'não identificado': 0
    }
    df_input['nivel_educacional'] = df_input['nivel_educacional'].map(mapa_educacional).fillna(0)

    for col in colunas_modelo:
        if col.startswith("tem_"):
            original = col.replace("tem_", "")
            df_input[col] = df_input[original].notnull().astype(int)
            df_input[original] = df_input[original].fillna(-9999)

    for col in colunas_modelo:
        if col not in df_input.columns:
            df_input[col] = 0

    return df_input[colunas_modelo]

def render():
    st.title("📝 Cadastro de Novo Candidato")

    estados = [
        "AC", "AL", "AP", "AM", "BA", "CE", "DF", "ES", "GO",
        "MA", "MT", "MS", "MG", "PA", "PB", "PR", "PE", "PI",
        "RJ", "RN", "RS", "RO", "RR", "SC", "SP", "SE", "TO"
    ]

    with st.form("form_candidato"):
        nome = st.text_input("Nome do candidato")
        idade = st.number_input("Idade", 16, 70, 30)
        estado = st.selectbox("Estado", estados)
        nivel_educacional = st.selectbox("Escolaridade", [
            "ensino fundamental", "ensino médio", "ensino superior",
            "superior incompleto", "pós-graduação ou mais", "não identificado"
        ])
        tempo_experiencia_anos = st.number_input("Tempo de experiência (anos)", 0, 50)
        quantidade_experiencias = st.number_input("Quantidade de experiências", 0, 30, 0)
        experiencia_sap = st.checkbox("Possui experiência com SAP?")
        remuneracao_mensal_brl = st.number_input("Remuneração mensal (R$)", value=0.0, step=100.0)

        submitted = st.form_submit_button("Classificar")

    if submitted:
        modelo, colunas_modelo = carregar_recursos()
        media, desvio = carregar_parametros_zscore()

        remuneracao_zscore = (remuneracao_mensal_brl - media) / desvio if desvio > 0 else 0

        input_dict = {
            "nome": nome,
            "estado": estado,
            "idade": idade,
            "nivel_educacional": nivel_educacional,
            "tempo_experiencia_anos": tempo_experiencia_anos,
            "quantidade_experiencias": quantidade_experiencias,
            "experiencia_sap": int(experiencia_sap),
            "remuneracao_mensal_brl": remuneracao_mensal_brl,
            "remuneracao_zscore": remuneracao_zscore  # necessário para o modelo
        }

        df_input = preparar_input(input_dict, colunas_modelo)
        cluster_predito = modelo.predict(df_input)[0]

        # Dicionário de nomes dos clusters
        nomes_clusters = {
            0: "Veteranos Invisíveis",
            1: "Exploradores em Branco",
            2: "Especialistas Aspiracionais",
            3: "Sombras do Cadastro"
        }

        nome_cluster = nomes_clusters.get(cluster_predito, f"Cluster {cluster_predito}")

        st.success(f"✅ O candidato foi classificado como pertencente ao **Cluster {cluster_predito} — {nome_cluster}**.")

        # Salvar no session_state
        st.session_state["candidato_classificado"] = {
            "cluster": cluster_predito,
            "nome_cluster": nome_cluster,
            "estado": estado,
            "nome": nome
        }

