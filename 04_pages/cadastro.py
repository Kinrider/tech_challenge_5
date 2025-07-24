import streamlit as st
import pandas as pd
import joblib

from transformadores_customizados import (
    HierarquiaOrdinalTransformer,
    EducacionalOrdinalTransformer,
    BooleanToInt
)

@st.cache_resource
def carregar_modelo_pipeline():
    modelo = joblib.load("03_modelos/modelo_kmeans.joblib")
    pipeline = joblib.load("03_modelos/pipeline_preprocessamento.joblib")
    colunas_modelo = joblib.load("03_modelos/colunas_usadas.joblib")
    return modelo, pipeline, colunas_modelo

def render():
    st.title("📝 Cadastro de Novo Candidato")

    estados = ["AC", "AL", "AP", "AM", "BA", "CE", "DF", "ES", "GO", "MA", "MT", "MS", "MG", "PA", "PB", "PR", "PE", "PI", "RJ", "RN", "RS", "RO", "RR", "SC", "SP", "SE", "TO"]
    niveis = ['Estagiário', 'Analista', 'Especialista', 'Consultor', 'Coordenador', 'Gerente', 'Diretor', 'C-Level']
    escolaridades = ['ensino fundamental', 'ensino médio', 'ensino superior', 'superior incompleto', 'pós-graduação ou mais', 'não identificado']
    categorias = [
        "Consultoria / Projetos", "Design / Criação", "Educação / Treinamento", "Engenharia",
        "Financeiro / Contábil", "Indefinido", "Jurídico", "Logística / Suprimentos",
        "Marketing / Comunicação", "RH / Pessoas", "Saúde", "Tecnologia da Informação"
    ]

    with st.form("form_candidato"):
        nome = st.text_input("Nome do candidato")
        estado = st.selectbox("Estado", estados)
        nivel_educacional = st.selectbox("Escolaridade", escolaridades)
        categoria_profissional = st.selectbox("Categoria Profissional", categorias)
        nivel_hierarquico = st.selectbox("Nível Hierárquico", niveis)
        experiencia_sap = st.checkbox("Possui experiência com SAP?")
        tempo_experiencia = st.number_input("Tempo de experiência (anos)", 0.0, 50.0, step=0.5)
        qtd_experiencias = st.number_input("Quantidade de experiências", 0, 50)
        remuneracao_zscore = st.number_input("Remuneração Z-Score (valor livre)")

        submitted = st.form_submit_button("Classificar")

    if submitted:
        modelo, pipeline, colunas_modelo = carregar_modelo_pipeline()

        input_dict = {
            "nivel_educacional": nivel_educacional,
            "nivel_hierarquico": nivel_hierarquico,
            "experiencia_sap": experiencia_sap,
            "tempo_experiencia_anos": tempo_experiencia,
            "quantidade_experiencias": qtd_experiencias,
            "remuneracao_zscore": remuneracao_zscore,
            "categoria_profissional": categoria_profissional
        }

        df_input = pd.DataFrame([input_dict])
        X_processado = pipeline.transform(df_input)

        df_final = pd.DataFrame(X_processado, columns=colunas_modelo)
        cluster_predito = modelo.predict(df_final)[0]

        nomes_clusters = {
            0: "Veteranos Invisíveis",
            1: "Exploradores em Branco",
            2: "Especialistas Aspiracionais",
            3: "Sombras do Cadastro"
        }

        nome_cluster = nomes_clusters.get(cluster_predito, f"Cluster {cluster_predito}")
        st.success(f"✅ O candidato foi classificado no **Cluster {cluster_predito} — {nome_cluster}**.")

if __name__ == "__main__":
    render()
