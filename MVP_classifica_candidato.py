import streamlit as st
import sys
from pathlib import Path

# Adiciona o diretório 04_pages ao path
sys.path.append(str(Path(__file__).parent / "04_pages"))

# Importa as páginas
from clusters import render as render_clusters
from cadastro import render as render_cadastro
from analise_regional import render as render_analise

# === Sidebar ===
st.sidebar.title("Tech Challenge 5")
aba = st.sidebar.radio("Navegação", ["📊 Clusters", "📝 Novo Candidato", "🌍 Análise Regional"])

# === Roteamento ===
if aba == "📊 Clusters":
    render_clusters()
elif aba == "📝 Novo Candidato":
    render_cadastro()
elif aba == "🌍 Análise Regional":
    render_analise()
