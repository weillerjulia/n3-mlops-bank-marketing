"""Treinamento e avaliação dos dois algoritmos candidatos (Modelo A x Modelo B)."""
from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (average_precision_score, f1_score, precision_score,
                             recall_score, roc_auc_score)
from sklearn.model_selection import StratifiedKFold
from sklearn.pipeline import Pipeline

from .features import SeletorFixo, VotingSelector
from .preprocessamento import criar_preprocessador

CLASSES = {"LogisticRegression": LogisticRegression, "RandomForestClassifier": RandomForestClassifier}


def criar_modelo(spec: dict, seed: int):
    cls = CLASSES[spec["classe"]]
    params = dict(spec.get("params", {}))
    params["random_state"] = seed
    return cls(**params)


def pipeline_modelo(features: list[str], spec: dict, seed: int) -> Pipeline:
    """Pipeline completo serializável: features eleitas + pré-processamento + modelo.
    Garante previsões consistentes: quem carrega o .joblib recebe tudo junto."""
    return Pipeline([
        ("selecao", SeletorFixo(features)),
        ("preprocessamento", criar_preprocessador()),
        ("modelo", criar_modelo(spec, seed)),
    ])


def calcular_metricas(y_true, proba, percentual_contato: float = 0.2, limiar: float = 0.5) -> dict:
    y_true = np.asarray(y_true)
    pred = (proba >= limiar).astype(int)
    prevalencia = float(y_true.mean())
    # Lift: quantas vezes a taxa de conversão no top X% supera ligar aleatoriamente
    n_top = max(1, int(len(proba) * percentual_contato))
    top = np.argsort(-proba)[:n_top]
    precisao_top = float(y_true[top].mean())
    return {
        "average_precision": float(average_precision_score(y_true, proba)),
        "roc_auc": float(roc_auc_score(y_true, proba)),
        "f1": float(f1_score(y_true, pred, zero_division=0)),
        "precision": float(precision_score(y_true, pred, zero_division=0)),
        "recall": float(recall_score(y_true, pred, zero_division=0)),
        "prevalencia": prevalencia,
        "conversao_top": precisao_top,
        "lift_top": precisao_top / prevalencia if prevalencia > 0 else float("nan"),
        "captura_top": float(y_true[top].sum() / max(1, y_true.sum())),
    }


def competir_cv(X: pd.DataFrame, y: pd.Series, cfg: dict) -> pd.DataFrame:
    """Validação cruzada estratificada no TREINO, com o mesmo protocolo para A e B.

    Em cada fold: (1) a votação de features é refeita só com o treino do fold,
    (2) A e B recebem exatamente o mesmo conjunto eleito, (3) ambos são avaliados
    no mesmo trecho de validação do fold. Nada do fold de validação vaza."""
    seed = cfg["projeto"]["seed"]
    f = cfg["features"]
    pct = cfg["negocio"]["percentual_contato"]
    skf = StratifiedKFold(n_splits=cfg["avaliacao"]["cv_folds"], shuffle=True, random_state=seed)
    linhas = []
    for fold, (i_tr, i_va) in enumerate(skf.split(X, y), start=1):
        X_tr, X_va, y_tr, y_va = X.iloc[i_tr], X.iloc[i_va], y.iloc[i_tr], y.iloc[i_va]
        if y_va.nunique() < 2 or y_tr.nunique() < 2:
            continue
        sel = VotingSelector(f["top_k"], f["votos_minimos"], f["min_features"], seed).fit(X_tr, y_tr)
        for chave in ("A", "B"):
            spec = cfg["modelos"][chave]
            pipe = pipeline_modelo(sel.features_eleitas_, spec, seed).fit(X_tr, y_tr)
            m = calcular_metricas(y_va, pipe.predict_proba(X_va)[:, 1], pct)
            linhas.append({"fold": fold, "modelo": chave, "algoritmo": spec["nome"],
                           "n_features": len(sel.features_eleitas_), **m})
    return pd.DataFrame(linhas)
