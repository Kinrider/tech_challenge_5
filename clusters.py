import streamlit as st

def render():
    st.title("📊 Perfil dos Clusters")
    st.markdown("""
    ## Análise de Clusterização de Candidatos
    Este relatório apresenta uma análise detalhada dos clusters obtidos a partir de um modelo de 
    KMeans aplicado sobre perfis de candidatos. A clusterização considerou dados educacionais, experiência, 
    presença de informações relevantes (como SAP e remuneração), e foi enriquecida com flags para dados ausentes.

    ### Clusters Identificados:

    ---
    **Cluster 0 — "Veteranos Invisíveis"**  
    🧠 Fortes tecnicamente, ocultam remuneração.  
    🏷️ Saúde | 🎓 Superior completo | 💼 Experiência presente | SAP: 32%

    ---
    **Cluster 1 — "Exploradores em Branco"**  
    🧠 Iniciantes ou perfis incompletos.  
    🏷️ TI | 🎓 Fundamental/médio incompleto | 💼 Sem experiência | SAP: 2%

    ---
    **Cluster 2 — "Especialistas Aspiracionais"**  
    🧠 Técnicos completos, alta qualificação.  
    🏷️ TI | 🎓 Pós/superior | 💼 Experiência alta | SAP: 35%

    ---
    **Cluster 3 — "Sombras do Cadastro"**  
    🧠 Perfis vazios ou abandonados.  
    🏷️ Saúde | 🎓 Médio | 💼 Sem experiência | SAP: 19%

    ---
    ### Considerações Finais:
    A clusterização permite identificar perfis distintos para ações estratégicas em recrutamento.
    """)
