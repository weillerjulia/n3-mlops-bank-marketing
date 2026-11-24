"""ETL: validação, limpeza, engenharia de atributos e pré-processamento.

Regra anti-vazamento: tudo que "aprende" com os dados (imputação, limites de
outliers, escala, codificação) fica dentro de um Pipeline/ColumnTransformer e é
ajustado apenas com dados de treino.
"""
from __future__ import annotations

import hashlib

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.compose import ColumnTransformer, make_column_selector
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

COLUNAS_ESPERADAS = [
    "age", "job", "marital", "education", "default", "housing", "loan", "contact",
    "month", "day_of_week", "duration", "campaign", "pdays", "previous", "poutcome",
    "emp.var.rate", "cons.price.idx", "cons.conf.idx", "euribor3m", "nr.employed", "y",
]
COLUNAS_CONTROLE = ["registro_id", "periodo"]


# ---------------------------------------------------------------- validação
def validar(df: pd.DataFrame, cfg: dict) -> dict:
    """Checagens de qualidade. Erros graves interrompem o pipeline (raise)."""
    faltando = [c for c in COLUNAS_ESPERADAS if c not in df.columns]
    if faltando:
        raise ValueError(f"Colunas ausentes na entrada: {faltando}")

    alvo = cfg["dados"]["alvo"]
    valores_alvo = set(df[alvo].dropna().unique())
    if not valores_alvo <= {"yes", "no"}:
        raise ValueError(f"Valores inesperados no alvo: {valores_alvo}")
    if (df["age"] < 17).any() or (df["age"] > 100).any():
        raise ValueError("Idades fora do intervalo plausível [17, 100].")
    if (df["campaign"] < 1).any():
        raise ValueError("campaign deve ser >= 1 (inclui o contato atual).")

    return {
        "linhas": int(len(df)),
        "duplicatas": int(df.drop(columns=COLUNAS_CONTROLE, errors="ignore").duplicated().sum()),
        "nulos": int(df.isna().sum().sum()),
        "unknown_por_coluna": {c: int((df[c] == "unknown").sum())
                               for c in df.select_dtypes("object").columns
                               if (df[c] == "unknown").any()},
        "taxa_positiva": float((df[alvo] == cfg["dados"]["classe_positiva"]).mean()),
    }


# ---------------------------------------------------------------- limpeza + atributos
def limpar_e_criar_atributos(df: pd.DataFrame, cfg: dict) -> pd.DataFrame:
    """Transformações determinísticas (não aprendem com os dados, logo não vazam)."""
    d = cfg["dados"]
    df = df.copy()

    # 1. Duplicatas exatas (mesmo cliente/contato registrado duas vezes)
    df = df[~df.drop(columns=COLUNAS_CONTROLE, errors="ignore").duplicated()]

    # 2. Remove colunas com vazamento (duration só existe após a ligação)
    df = df.drop(columns=d["colunas_removidas"])

    # 3. Alvo binário
    df[d["alvo"]] = (df[d["alvo"]] == d["classe_positiva"]).astype(int)

    # 4. Engenharia de atributos
    #    pdays = 999 significa "nunca contatado" -> número mágico que distorce a escala
    df["contatado_antes"] = (df["pdays"] != 999).astype(int)
    df["pdays"] = df["pdays"].where(df["pdays"] != 999, np.nan)  # imputado no pipeline
    #    Intensidade de contato na campanha atual vs. histórico
    df["contatos_totais"] = df["campaign"] + df["previous"]
    #    Faixa etária: relação não linear entre idade e adesão (jovens e aposentados aderem mais)
    df["faixa_etaria"] = pd.cut(df["age"], bins=[0, 25, 35, 45, 55, 65, 120],
                                labels=["<=25", "26-35", "36-45", "46-55", "56-65", "65+"]).astype(str)

    # 5. "unknown" é mantido como categoria: a ausência de informação é informativa
    return df.reset_index(drop=True)


def separar_xy(df: pd.DataFrame, cfg: dict):
    alvo = cfg["dados"]["alvo"]
    X = df.drop(columns=[alvo, *COLUNAS_CONTROLE], errors="ignore")
    return X, df[alvo]


def particao_do_registro(registro_id: int, seed: int) -> float:
    """Número pseudoaleatório em [0, 1) determinado só pelo ID do registro."""
    h = hashlib.md5(f"{seed}-{registro_id}".encode()).hexdigest()
    return int(h, 16) % 10_000 / 10_000


def particionar(df: pd.DataFrame, cfg: dict):
    """Treino | validação | teste estáveis entre ciclos (hash do registro_id).

    Um contato que caiu na validação no ciclo 1 continua na validação em todos
    os ciclos seguintes. Isso permite comparar o modelo vigente (treinado em um
    ciclo anterior) com o candidato na mesma validação, sem que o vigente tenha
    visto esses registros no treino."""
    a = cfg["avaliacao"]
    u = df["registro_id"].map(lambda i: particao_do_registro(i, cfg["projeto"]["seed"]))
    lim_treino = 1 - a["proporcao_validacao"] - a["proporcao_teste"]
    lim_val = 1 - a["proporcao_teste"]
    treino = df[u < lim_treino]
    val = df[(u >= lim_treino) & (u < lim_val)]
    teste = df[u >= lim_val]
    return treino.reset_index(drop=True), val.reset_index(drop=True), teste.reset_index(drop=True)


# ---------------------------------------------------------------- transformadores
class LimitadorOutliers(BaseEstimator, TransformerMixin):
    """Winsorização: limita valores aos percentis aprendidos NO TREINO."""

    def __init__(self, q_inf: float = 0.01, q_sup: float = 0.99):
        self.q_inf = q_inf
        self.q_sup = q_sup

    def fit(self, X, y=None):
        X = np.asarray(X, dtype=float)
        self.inf_ = np.nanquantile(X, self.q_inf, axis=0)
        self.sup_ = np.nanquantile(X, self.q_sup, axis=0)
        return self

    def transform(self, X):
        return np.clip(np.asarray(X, dtype=float), self.inf_, self.sup_)

    def get_feature_names_out(self, input_features=None):
        return np.asarray(input_features, dtype=object)


def criar_preprocessador() -> ColumnTransformer:
    """Numéricas: imputação (mediana) -> limite de outliers -> padronização.
    Categóricas: imputação (moda) -> one-hot (ignora categorias novas em produção).
    As colunas são escolhidas por tipo, então o mesmo objeto serve para qualquer
    subconjunto de features eleito na votação."""
    numericas = Pipeline([
        # keep_empty_features: em 2008 ninguém tinha campanha anterior, então
        # pdays chega 100% vazio no lote inicial; a coluna é mantida (preenchida
        # com 0) para que o esquema seja estável entre ciclos.
        ("imputador", SimpleImputer(strategy="median", keep_empty_features=True)),
        ("outliers", LimitadorOutliers()),
        ("escala", StandardScaler()),
    ])
    categoricas = Pipeline([
        ("imputador", SimpleImputer(strategy="most_frequent")),
        ("onehot", OneHotEncoder(handle_unknown="ignore", sparse_output=False)),
    ])
    return ColumnTransformer([
        ("num", numericas, make_column_selector(dtype_include=np.number)),
        ("cat", categoricas, make_column_selector(dtype_exclude=np.number)),
    ], verbose_feature_names_out=True)
