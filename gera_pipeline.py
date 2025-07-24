import pandas as pd
import joblib
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, OneHotEncoder

from transformadores_customizados import (
    HierarquiaOrdinalTransformer,
    EducacionalOrdinalTransformer,
    BooleanToInt
)

# === Definição de colunas ===
colunas_ordinal = ["nivel_hierarquico", "nivel_educacional"]
colunas_booleanas = ["experiencia_sap"]
colunas_numericas = ["remuneracao_zscore", "tempo_experiencia_anos", "quantidade_experiencias"]
colunas_categoricas = ["categoria_profissional"]

# === Pipelines por tipo ===
ordinal_pipeline = ColumnTransformer(transformers=[
    ("hierarquia", HierarquiaOrdinalTransformer(), "nivel_hierarquico"),
    ("educacional", EducacionalOrdinalTransformer(), "nivel_educacional")
])

boolean_pipeline = Pipeline([
    ("bool_to_int", BooleanToInt())
])

numerico_pipeline = Pipeline([
    ("imputer", SimpleImputer(strategy="median")),
    ("scaler", StandardScaler())
])

# === Compor pipeline completo ===
preprocessador = ColumnTransformer(transformers=[
    ("ordinais", ordinal_pipeline, colunas_ordinal),
    ("booleanos", boolean_pipeline, colunas_booleanas),
    ("numericos", numerico_pipeline, colunas_numericas),
    ("onehot", OneHotEncoder(handle_unknown="ignore", drop="first"), colunas_categoricas)
])

# === Salvar pipeline ===
joblib.dump(preprocessador, "03_modelos/pipeline_preprocessamento.joblib")
print("✅ Pipeline salvo em: 03_modelos/pipeline_preprocessamento.joblib")
