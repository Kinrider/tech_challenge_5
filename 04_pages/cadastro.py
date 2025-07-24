import streamlit as st
import pandas as pd
import joblib
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
import requests
from io import BytesIO

@st.cache_resource
def carregar_modelo_e_scaler():
    BASE_GITHUB = "https://github.com/Kinrider/tech_challenge_5/raw/main/03_modelos"
    modelo = joblib.load(BytesIO(requests.get(f"{BASE_GITHUB}/modelo_kmeans.joblib").content))
    colunas_modelo = joblib.load(BytesIO(requests.get(f"{BASE_GITHUB}/colunas_usadas.joblib").content))
    return modelo, colunas_modelo

def aplicar_tratamentos(df):
    # Tratamento de nivel_hierarquico
    mapa_hierarquico = {
        'estagiario': 1, 'analista': 2, 'especialista': 3, 'consultor': 3,
        'coordenador': 4, 'gerente': 5, 'diretor': 6, 'c-level': 7,
        'não identificado': 0, None: 0
    }
    df['nivel_hierarquico'] = df['nivel_hierarquico'].str.lower().fillna("não identificado")
    df['nivel_hierarquico'] = df['nivel_hierarquico'].map(mapa_hierarquico).fillna(0).astype(int)

    # Tratamento de nivel_educacional
    mapa_educacional = {
        'ensino fundamental': 1, 'ensino médio': 2, 'ensino superior': 3,
        'superior incompleto': 3, 'pós-graduação ou mais': 4,
        'não identificado': 0, None: 0
    }
    df['nivel_educacional'] = df['nivel_educacional'].map(mapa_educacional).fillna(0).astype(int)

    # Conversão de variável booleana experiencia_sap
    df['experiencia_sap'] = df['experiencia_sap'].astype(int)

    # One-hot encoding para variáveis categóricas
    df = pd.get_dummies(df, columns=['categoria_profissional'], drop_first=True)

    # Variáveis contínuas + tratamento com pipeline interno
    variaveis_continuas = ['remuneracao_zscore', 'tempo_experiencia_anos', 'quantidade_experiencias']
    pipeline_numerico = Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler())
    ])
    df[variaveis_continuas] = pipeline_numerico.fit_transform(df[variaveis_continuas])

    # Flags de presença
    for col in variaveis_continuas:
        df[f"tem_{col}"] = df[col].notnull().astype(int)

    return df

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
        modelo, colunas_modelo = carregar_modelo_e_scaler()

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
        df_tratado = aplicar_tratamentos(df_input)

        for col in colunas_modelo:
            if col not in df_tratado.columns:
                df_tratado[col] = 0
        df_final = df_tratado[colunas_modelo]

        cluster_predito = modelo.predict(df_final)[0]

        nomes_clusters = {
            0: "Exploradores Técnicos",
            1: "Veteranos Invisíveis",
            2: "Sombras do Cadastro",
            3: "Especialistas Aspiracionais"
        }

        nome_cluster = nomes_clusters.get(cluster_predito, f"Cluster {cluster_predito}")
        st.success(f"✅ O candidato foi classificado no **Cluster {cluster_predito} — {nome_cluster}**.")
        
        # Salva no session_state
        st.session_state["candidato_classificado"] = {
            "nome": nome,
            "estado": estado,
            "cluster": cluster_predito
        }

if __name__ == "__main__":
    render()
