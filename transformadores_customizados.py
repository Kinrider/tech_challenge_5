from sklearn.base import BaseEstimator, TransformerMixin

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
