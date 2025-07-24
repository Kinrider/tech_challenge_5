import pandas as pd
import numpy as np
import joblib
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
import os

# === Transformadores personalizados ===
class HierarquiaOrdinalTransformer(BaseEstimator, TransformerMixin):
    def __init__(self):
        self.mapa = {
            'estagiario': 1, 'analista': 2, 'especialista': 3, 'consultor': 3,
            'coordenador': 4, 'gerente': 5, 'diretor': 6, 'c-level': 7,
            'não identificado': 0, None: 0
        }

    def fit(self, X, y=None): return self
    def transform(self, X):
        return X.fillna("não identificado").str.lower().map(self.mapa).fillna(0).astype(int).to_frame()

class EducacionalOrdinalTransformer(BaseEstimator, TransformerMixin):
    def __init__(self):
        self.mapa = {
            'ensino fundamental': 1, 'ensino médio': 2, 'ensino superior': 3,
            'superior incompleto': 3, 'pós-graduação ou mais': 4,
            'não identificado': 0, None: 0
        }

    def fit(self, X, y=None): return self
    def transform(self, X):
        return X.map(self.mapa).fillna(0).astype(int).to_frame()

class BooleanToInt(BaseEstimator, TransformerMixin):
    def fit(self, X, y=None): return self
    def transform(self, X): return X.astype(int)

# === Definição das colunas do modelo ===
colunas_numericas = ['remuneracao_zscore', 'tempo_experiencia_anos', 'quantidade_experiencias']
coluna_hierarquia = ['nivel_hierarquico']
coluna_educacional = ['nivel_educacional']
coluna_sap = ['experiencia_sap']
coluna_categoria = ['categoria_profissional']

# === Pipelines ===
pipeline_num = Pipeline([
    ("imputer", SimpleImputer(strategy="median")),
    ("scaler", StandardScaler())
])

pipeline_sap = Pipeline([
    ("bool_to_int", BooleanToInt())
])

pipeline_cat = Pipeline([
    ("onehot", OneHotEncoder(drop="first", sparse=False, handle_unknown='ignore'))
])

# === Pipeline completo ===
preprocessador = ColumnTransformer([
    ("num", pipeline_num, colunas_numericas),
    ("hierarquia", HierarquiaOrdinalTransformer(), coluna_hierarquia),
    ("educacional", EducacionalOrdinalTransformer(), coluna_educacional),
    ("sap", pipeline_sap, coluna_sap),
    ("categoria", pipeline_cat, coluna_categoria)
])

# === Salvamento final ===
caminho_saida = r"C:\Users\pedro\Documents\Área de Trabalho\tech_challenge_5\03_modelos"
os.makedirs(caminho_saida, exist_ok=True)
joblib.dump(preprocessador, os.path.join(caminho_saida, "pipeline_preprocessamento.joblib"))

print("✅ Pipeline salvo com sucesso em:", caminho_saida)
