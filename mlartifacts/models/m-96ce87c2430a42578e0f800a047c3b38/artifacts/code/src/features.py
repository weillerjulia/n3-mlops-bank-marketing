"""Três métodos de importância/seleção de atributos + votação.

Os métodos trabalham sobre a matriz pré-processada (one-hot), mas a votação é
feita no nível das FEATURES ORIGINAIS: as importâncias das colunas one-hot de uma
mesma variável (ex.: job_admin., job_retired...) são agregadas. Assim, o conjunto
eleito é interpretável ("job", "euribor3m") e o modelo final recebe a variável
inteira.

O VotingSelector é um transformador do scikit-learn: dentro da validação cruzada
ele é reajustado em cada fold (sem vazamento), e no treino completo produz o
conjunto eleito do ciclo.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_selection import RFE, mutual_info_classif
from sklearn.linear_model import LogisticRegression

from .preprocessamento import criar_preprocessador


def _mapear_para_originais(nomes_codificados, colunas_originais):
    """'num__age' -> 'age'; 'cat__job_admin.' -> 'job' (prefixo mais longo)."""
    ordenadas = sorted(colunas_originais, key=len, reverse=True)
    mapa = []
    for nome in nomes_codificados:
        bloco, resto = nome.split("__", 1)
        if bloco == "num":
            mapa.append(resto)
        else:
            mapa.append(next(c for c in ordenadas if resto.startswith(c + "_")))
    return np.array(mapa)


class VotingSelector(BaseEstimator, TransformerMixin):
    def __init__(self, top_k=8, votos_minimos=2, min_features=6, seed=42):
        self.top_k = top_k
        self.votos_minimos = votos_minimos
        self.min_features = min_features
        self.seed = seed

    # -------------------------------------------------- métodos individuais
    def _selectkbest(self, Xc, y, discretas):
        # SelectKBest com informação mútua: captura relações não lineares e
        # funciona para variáveis numéricas e binárias (one-hot).
        return mutual_info_classif(Xc, y, discrete_features=discretas, random_state=self.seed)

    def _random_forest(self, Xc, y):
        rf = RandomForestClassifier(n_estimators=150, max_depth=10, min_samples_leaf=20,
                                    class_weight="balanced_subsample", n_jobs=-1,
                                    random_state=self.seed)
        rf.fit(Xc, y)
        return rf.feature_importances_

    def _rfe(self, Xc, y):
        # RFE com Regressão Logística (dados padronizados): elimina 10% das
        # colunas por rodada até sobrar 1 e devolve o ranking completo.
        est = LogisticRegression(max_iter=1000, class_weight="balanced", solver="lbfgs")
        rfe = RFE(est, n_features_to_select=1, step=0.1)
        rfe.fit(Xc, y)
        return rfe.ranking_

    # -------------------------------------------------- fit / transform
    def fit(self, X: pd.DataFrame, y):
        y = np.asarray(y)
        self.colunas_entrada_ = list(X.columns)
        prep = criar_preprocessador()
        Xc = prep.fit_transform(X, y)  # ajustado só com os dados recebidos (treino/fold)
        nomes = prep.get_feature_names_out()
        originais = _mapear_para_originais(nomes, self.colunas_entrada_)
        discretas = np.array([n.startswith("cat__") for n in nomes])

        mi = self._selectkbest(Xc, y, discretas)
        imp = self._random_forest(Xc, y)
        rank_rfe = self._rfe(Xc, y)

        agg = pd.DataFrame({"feature": originais, "mi": mi, "rf": imp, "rfe": rank_rfe})
        agg = agg.groupby("feature").agg(
            selectkbest_score=("mi", "max"),          # melhor coluna da variável
            random_forest_importance=("rf", "sum"),   # importância total da variável
            rfe_melhor_posicao=("rfe", "min"),        # coluna da variável que sobreviveu mais
        )

        agg["rank_selectkbest"] = agg["selectkbest_score"].rank(ascending=False, method="min").astype(int)
        agg["rank_random_forest"] = agg["random_forest_importance"].rank(ascending=False, method="min").astype(int)
        agg["rank_rfe"] = agg["rfe_melhor_posicao"].rank(ascending=True, method="min").astype(int)

        metodos = ["selectkbest", "random_forest", "rfe"]
        for m in metodos:
            agg[f"voto_{m}"] = (agg[f"rank_{m}"] <= self.top_k).astype(int)
        agg["votos"] = agg[[f"voto_{m}" for m in metodos]].sum(axis=1)
        agg["soma_rankings"] = agg[[f"rank_{m}" for m in metodos]].sum(axis=1)

        # Regra: votos >= votos_minimos; desempate e ordenação pela soma de rankings
        agg = agg.sort_values(["votos", "soma_rankings"], ascending=[False, True])
        eleitas = agg.index[agg["votos"] >= self.votos_minimos].tolist()
        if len(eleitas) < self.min_features:  # completa até o mínimo pela soma de rankings
            extras = [f for f in agg.index if f not in eleitas][: self.min_features - len(eleitas)]
            eleitas += extras
        agg["eleita"] = agg.index.isin(eleitas)

        self.tabela_votos_ = agg.reset_index()
        self.features_eleitas_ = eleitas
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        return X[self.features_eleitas_]

    def get_feature_names_out(self, input_features=None):
        return np.asarray(self.features_eleitas_, dtype=object)


class SeletorFixo(BaseEstimator, TransformerMixin):
    """Seleciona um conjunto de features já eleito (usado no modelo final)."""

    def __init__(self, features=None):
        self.features = features

    def fit(self, X, y=None):
        faltando = [f for f in self.features if f not in X.columns]
        if faltando:
            raise ValueError(f"Features eleitas ausentes na entrada: {faltando}")
        return self

    def transform(self, X):
        return X[list(self.features)]

    def get_feature_names_out(self, input_features=None):
        return np.asarray(self.features, dtype=object)
