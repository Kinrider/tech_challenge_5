import streamlit as st
import pandas as pd
import joblib
import requests
from io import BytesIO

@st.cache_resource
def carregar_recursos():
    BASE_GITHUB = "https://github.com/Kinrider/tech_challenge_5/raw/main/03_modelos"
    modelo = joblib.load(BytesIO(requests.get(f"{BASE_GITHUB}/modelo_kmeans.joblib").content))
    colunas_modelo = joblib.load(BytesIO(requests.get(f"{BASE_GITHUB}/colunas_usadas.joblib").content))
    scaler = joblib.load(BytesIO(requests.get(f"{BASE_GITHUB}/scaler.joblib").content))
    return modelo, colunas_modelo, scaler

@st.cache_data
def carregar_parametros_zscore():
    df = pd.read_parquet(
        'https://github.com/Kinrider/tech_challenge_5/raw/refs/heads/main/01_fontes/arquivos_decision/fontes_tratadas/remuneracao_mensal_only.parquet',
        engine="pyarrow"
    )
    media = df["remuneracao_mensal_brl"].replace(-9999, pd.NA).dropna().mean()
    desvio = df["remuneracao_mensal_brl"].replace(-9999, pd.NA).dropna().std()
    return media, desvio

def preparar_input(form_dict, colunas_modelo, scaler):
    df_input = pd.DataFrame([form_dict])

    # === Mapeamento de escolaridade ===
    mapa_educacional = {
        'ensino fundamental': 1,
        'ensino médio': 2,
        'ensino superior': 3,
        'superior incompleto': 3,
        'pós-graduação ou mais': 4,
        'não identificado': 0,
        None: 0
    }
    df_input['nivel_educacional'] = df_input['nivel_educacional'].map(mapa_educacional).fillna(0).astype(int)

    # === Conversão booleana ===
    df_input['experiencia_sap'] = df_input['experiencia_sap'].astype(int)

    # === One-hot encoding ===
    colunas_categoricas = ['categoria_profissional', 'subarea_profissional', 'nivel_hierarquico']
    df_input = pd.get_dummies(df_input, columns=colunas_categoricas, drop_first=True)

    # === Tratamento de NaNs ===
    colunas_com_nan = df_input.columns[df_input.isnull().any()].tolist()
    for col in colunas_com_nan:
        df_input[f"tem_{col}"] = df_input[col].notnull().astype(int)
        df_input[col] = df_input[col].fillna(-9999)

    # === Escalonamento (usando scaler salvo) ===
    variaveis_continuas = ['remuneracao_zscore', 'tempo_experiencia_anos', 'quantidade_experiencias']
    df_input[variaveis_continuas] = scaler.transform(df_input[variaveis_continuas])

    # === Garantir colunas compatíveis com o modelo ===
    for col in colunas_modelo:
        if col not in df_input.columns:
            df_input[col] = 0

    return df_input[colunas_modelo]

def render():
    st.title("📝 Cadastro de Novo Candidato")

    estados = [
        "AC", "AL", "AP", "AM", "BA", "CE", "DF", "ES", "GO", "MA", "MT", "MS", "MG", "PA", "PB",
        "PR", "PE", "PI", "RJ", "RN", "RS", "RO", "RR", "SC", "SP", "SE", "TO"
    ]
    categorias = ['Comercial / Negócios', 'Consultoria / Projetos', 'Design / Criação', 'Educação / Treinamento', 'Engenharia', 'Financeiro / Contábil', 'Indefinido', 'Jurídico', 'Logística / Suprimentos', 'Marketing / Comunicação', 'RH / Pessoas', 'Saúde', 'Tecnologia da Informação']
    subareas = ['Análise de Dados', 'Contabilidade', 'Controladoria', 'Desenho Técnico', 'Desenvolvimento Backend', 'Desenvolvimento Frontend', 'Desenvolvimento Full Stack', 'Engenharia de Dados', 'Gestão de Produtos', 'Gestão de Projetos Ágeis', 'Indefinido', 'Infraestrutura de TI', 'Marketing Digital', 'Qualidade de Software / QA', 'Recrutamento e Seleção', 'Segurança da Informação', 'Sistemas Corporativos SAP / ERP / CRM', 'UX/UI Design', 'Vendas']
    niveis = ['Analista', 'C-Level', 'Consultor', 'Coordenador', 'Diretor', 'Especialista', 'Estagiário', 'Gerente', 'Indefinido', 'Júnior', 'Líder', 'Pleno', 'Sênior', 'Trainee', 'Técnico']

    with st.form("form_candidato"):
        nome = st.text_input("Nome do candidato")
        idade = st.number_input("Idade", 16, 70, 30)
        estado = st.selectbox("Estado", estados)
        nivel_educacional = st.selectbox("Escolaridade", [
            "ensino fundamental", "ensino médio", "ensino superior",
            "superior incompleto", "pós-graduação ou mais", "não identificado"
        ])
        categoria = st.selectbox("Categoria Profissional", categorias)
        subarea = st.selectbox("Subárea Profissional", subareas)
        nivel_hierarquico = st.selectbox("Nível Hierárquico", niveis)
        tempo_experiencia_anos = st.number_input("Tempo de experiência (anos)", 0, 50)
        quantidade_experiencias = st.number_input("Quantidade de experiências", 0, 30, 0)
        experiencia_sap = st.checkbox("Possui experiência com SAP?")
        remuneracao_mensal_brl = st.number_input("Remuneração mensal (R$)", value=0.0, step=100.0)

        submitted = st.form_submit_button("Classificar")

    if submitted:
        modelo, colunas_modelo, scaler = carregar_recursos()
        media, desvio = carregar_parametros_zscore()
        remuneracao_zscore = (remuneracao_mensal_brl - media) / desvio if desvio > 0 else 0

        input_dict = {
            "nome": nome,
            "estado": estado,
            "idade": idade,
            "nivel_educacional": nivel_educacional,
            "categoria_profissional": categoria,
            "subarea_profissional": subarea,
            "nivel_hierarquico": nivel_hierarquico,
            "tempo_experiencia_anos": tempo_experiencia_anos,
            "quantidade_experiencias": quantidade_experiencias,
            "experiencia_sap": experiencia_sap,
            "remuneracao_mensal_brl": remuneracao_mensal_brl,
            "remuneracao_zscore": remuneracao_zscore
        }

        df_input = preparar_input(input_dict, colunas_modelo, scaler)
        cluster_predito = modelo.predict(df_input)[0]
        st.write("Input ao modelo:", df_input)

        nomes_clusters = {
            0: "Veteranos Invisíveis",
            1: "Exploradores em Branco",
            2: "Especialistas Aspiracionais",
            3: "Sombras do Cadastro"
        }

        nome_cluster = nomes_clusters.get(cluster_predito, f"Cluster {cluster_predito}")
        st.success(f"✅ O candidato foi classificado como pertencente ao **Cluster {cluster_predito} — {nome_cluster}**.")

        st.session_state["candidato_classificado"] = {
            "cluster": cluster_predito,
            "nome_cluster": nome_cluster,
            "estado": estado,
            "nome": nome
        }
