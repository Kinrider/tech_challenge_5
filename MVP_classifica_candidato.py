import streamlit as st
import pandas as pd
import joblib
import requests
from io import BytesIO
import numpy as np

# === URLs do seu repositório GitHub (branch main) ===
BASE_GITHUB = "https://github.com/Kinrider/tech_challenge_5/raw/refs/heads/main"

URL_MODELO = f"{BASE_GITHUB}/03_modelos/modelo_kmeans.joblib"
URL_COLUNAS = f"{BASE_GITHUB}/03_modelos/colunas_usadas.joblib"
URL_CLUSTERIZADOS = f"{BASE_GITHUB}/03_modelos/04_clusterizados.parquet"
URL_CANDIDATOS_ORIGINAIS = "https://github.com/Kinrider/tech_challenge_5/raw/main/01_fontes/arquivos_decision/fontes_tratadas/01_candidatos.parquet"

# === Carregar arquivos remoto ===
modelo = joblib.load(BytesIO(requests.get(URL_MODELO).content))
colunas_modelo = joblib.load(BytesIO(requests.get(URL_COLUNAS).content))
df_clusterizados = pd.read_parquet(URL_CLUSTERIZADOS, engine="pyarrow")
df_candidatos = pd.read_parquet(URL_CANDIDATOS_ORIGINAIS, engine="pyarrow")

# === Sidebar ===
st.sidebar.title("Tech Challenge 5")
aba = st.sidebar.radio("Navegação", ["📊 Clusters", "📝 Novo Candidato", "🌍 Análise Regional"])

# === Função auxiliar: tratamento do input do candidato ===
def preparar_input(form_dict):
    df_input = pd.DataFrame([form_dict])

    # Tratamento da escolaridade
    mapa_educacional = {
        'ensino fundamental': 1,
        'ensino médio': 2,
        'ensino superior': 2,
        'superior incompleto': 2,
        'pós-graduação ou mais': 3,
        'não identificado': 0
    }
    df_input['nivel_educacional'] = df_input['nivel_educacional'].map(mapa_educacional).fillna(0)

    # Flags e sentinelas
    for col in colunas_modelo:
        if col.startswith("tem_"):
            original = col.replace("tem_", "")
            df_input[col] = df_input[original].notnull().astype(int)
            df_input[original] = df_input[original].fillna(-9999)

    # Garantir todas as colunas do modelo
    for col in colunas_modelo:
        if col not in df_input.columns:
            df_input[col] = 0

    return df_input[colunas_modelo]

# === Aba 1: Relatório dos Clusters ===
if aba == "📊 Clusters":
    st.title("📊 Perfil dos Clusters")
    st.markdown("""
                      
    ## Análise de Clusterização de Candidatos
    Este relatório apresenta uma análise detalhada dos clusters obtidos a partir de um modelo de 
    KMeans aplicado sobre perfis de candidatos. A clusterização considerou dados educacionais, experiência, 
    presença de informações relevantes (como SAP e remuneração), e foi enriquecida com flags para dados ausentes.
                  
                      
    ### Clusters Identificados:
    
     ---
    
    **Cluster 0 — "Veteranos Invisíveis"**  

    🧠 **Descrição**:  
    
    Profissionais com forte formação e vivência, mas que preferem manter  
    discrição em relação à remuneração e objetivos. Muitos ocupam posições de gerência.  
    Perfil técnico e maduro, provavelmente estável no mercado.  
    
    🏷️ **Área mais frequente**: Saúde  
    💰 **Remuneração**: não informada  
    🎓 **Escolaridade**: ensino superior completo  
    💼 **Experiência**: média positiva (experiência presente)  
    🧠 **Experiência com SAP**: 32% possuem experiência

    ---

    **Cluster 1 — "Exploradores em Branco"**  

    🧠 **Descrição**:  
    
    Candidatos iniciantes ou com perfis incompletos. Representam possível  
    público jovem, sem trajetória definida, ou registros abandonados. Baixo engajamento com o sistema.  

    🏷️ **Área mais frequente**: Tecnologia da Informação  
    💰 **Remuneração**: ligeiramente abaixo da média  
    🎓 **Escolaridade**: ensino fundamental/médio incompleto  
    💼 **Experiência**: ausente  
    🧠 **Experiência com SAP**: 2% possuem experiência  
    
    ---

    **Cluster 2 — "Especialistas Aspiracionais"**  

    🧠 **Descrição**:  
    
    Grupo mais qualificado e competitivo. Profissionais com alta formação e vivência técnica,  
    geralmente de TI. São os candidatos mais completos e com maior potencial para posições de liderança  
    ou alta performance.  

    🏷️ **Área mais frequente**: Tecnologia da Informação  
    💰 **Remuneração**: acima da média  
    🎓 **Escolaridade**: pós-graduação ou superior completo  
    💼 **Experiência**: alta  
    🧠 **Experiência com SAP**: 35% possuem experiência  
    
    ---

    **Cluster 3 — "Sombras do Cadastro"**  

    🧠 **Descrição**:
      
    Usuários com perfis extremamente vazios. Podem representar registros não finalizados  
    ou abandonados. Pouca utilidade em campanhas de contratação até que o preenchimento seja refeito.    

    🏷️ **Área mais frequente**: Saúde  
    💰 **Remuneração**: não informada  
    🎓 **Escolaridade**: ensino médio  
    💼 **Experiência**: ausente  
    🧠 **Experiência com SAP**: 19% possuem experiência  

    ---
    
    ### Considerações Finais:
    
    A clusterização permitiu identificar perfis bem distintos, desde profissionais completos e preparados para o mercado até cadastros escassos de informações. A presença ou ausência de variáveis-chave (como remuneração e experiência) foi crucial para a separação dos grupos. As informações obtidas podem apoiar campanhas de recrutamento, filtragem de perfis, e ações para enriquecer cadastros incompletos.
    
    """)

# === Aba 2: Cadastro e Classificação ===
elif aba == "📝 Novo Candidato":
    st.title("📝 Cadastro de Novo Candidato")

    with st.form("form_candidato"):
        nome = st.text_input("Nome do candidato")
        idade = st.number_input("Idade", 16, 70, 30)
        municipio = st.text_input("Município")
        nivel_educacional = st.selectbox("Escolaridade", [
            "ensino fundamental", "ensino médio", "ensino superior",
            "superior incompleto", "pós-graduação ou mais", "não identificado"
        ])
        tempo_experiencia_anos = st.number_input("Tempo de experiência (anos)", 0.0, 50.0, step=0.5)
        quantidade_experiencias = st.number_input("Quantidade de experiências", 0, 30, 0)
        experiencia_sap = st.checkbox("Possui experiência com SAP?")
        remuneracao_zscore = st.number_input("Remuneração (z-score)", value=0.0)

        submitted = st.form_submit_button("Classificar")

    if submitted:
        input_dict = {
            "nome": nome,
            "municipio": municipio,
            "idade": idade,
            "nivel_educacional": nivel_educacional,
            "tempo_experiencia_anos": tempo_experiencia_anos,
            "quantidade_experiencias": quantidade_experiencias,
            "experiencia_sap": int(experiencia_sap),
            "remuneracao_zscore": remuneracao_zscore
        }

        df_input = preparar_input(input_dict)
        cluster_predito = modelo.predict(df_input)[0]

        st.success(f"✅ O candidato foi classificado como pertencente ao **Cluster {cluster_predito}**.")
        st.session_state["candidato_classificado"] = {
            "cluster": cluster_predito,
            "municipio": municipio,
            "nome": nome
        }

# === Aba 3: Análise Regional ===
elif aba == "🌍 Análise Regional":
    st.title("🌍 Análise Regional do Candidato")

    candidato = st.session_state.get("candidato_classificado")
    if not candidato:
        st.warning("⚠️ Classifique um candidato primeiro na aba anterior.")
    else:
        cluster = candidato["cluster"]
        municipio = candidato["municipio"]
        nome = candidato["nome"]

        st.subheader(f"📍 Perfil dos candidatos em **{municipio}** do **Cluster {cluster}**")

        df_filtrado = df_candidatos[
            (df_clusterizados["cluster"] == cluster) &
            (df_candidatos["municipio"].str.lower() == municipio.lower())
        ]

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

            # 🔄 Aqui futuramente você pode adicionar um mapa com a densidade geográfica

