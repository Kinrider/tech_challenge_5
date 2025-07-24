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

# === Carregar base tratada ===
df = pd.read_parquet("03_modelos/dados_input_modelo.parquet")

# === Definição de colunas ===
colunas_ordinal = ["nivel_hierarquico", "nivel_educacional"]
colunas_booleanas = ["experiencia_sap"]
colunas_numericas = ["remuneracao_zscore", "tempo_experiencia_anos", "quantidade_experiencias"]
colunas_categoricas = ["categoria_profissional"]

# === Pipelines por tipo ===
ordinal_pipeline = Pipeline([
    ("hierarquia", HierarquiaOrdinalTransformer())
])

educacional_pipeline = Pipeline([
    ("educacional", EducacionalOrdinalTransformer())
])

boolean_pipeline = Pipeline([
    ("bool_to_int", BooleanToInt())
])

numerico_pipeline = Pipeline([
    ("imputer", SimpleImputer(strategy="median")),
    ("scaler", StandardScaler())
])

# === Compor ColumnTransformer final ===
preprocessador = ColumnTransformer(transformers=[
    ("nivel_hierarquico", HierarquiaOrdinalTransformer(), "nivel_hierarquico"),
    ("nivel_educacional", EducacionalOrdinalTransformer(), "nivel_educacional"),
    ("experiencia_sap", BooleanToInt(), ["experiencia_sap"]),
    ("numericos", numerico_pipeline, colunas_numericas),
    ("categoria_profissional", OneHotEncoder(handle_unknown="ignore", drop="first"), colunas_categoricas)
])

# === Treinar pipeline com o DataFrame ===
preprocessador.fit(df)

# === Salvar pipeline treinado ===
joblib.dump(preprocessador, "03_modelos/pipeline_preprocessamento.joblib")
print("✅ Pipeline treinado e salvo com sucesso em: 03_modelos/pipeline_preprocessamento.joblib")
