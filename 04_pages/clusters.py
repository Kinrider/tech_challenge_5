import streamlit as st

def render():
    st.title("📊 Perfil dos Clusters")
    st.markdown("""
    ## Análise de Clusterização de Candidatos
    Este relatório apresenta uma análise detalhada dos clusters obtidos a partir de um modelo de KMeans aplicado sobre perfis de candidatos. A clusterização considerou dados educacionais, experiência, presença de informações relevantes (como SAP e remuneração), e foi enriquecida com flags para dados ausentes.

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
