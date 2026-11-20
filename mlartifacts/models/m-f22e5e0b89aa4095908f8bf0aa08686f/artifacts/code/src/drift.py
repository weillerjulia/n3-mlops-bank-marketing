"""Critério 10 - Monitoramento de drift com alerta.

Problema: os dados da campanha mudam muito ao longo do tempo (crise de 2008,
queda da Euribor, mudança no perfil de clientes). Um modelo treinado em 2008
pode ficar desatualizado sem que ninguém perceba.

Solução: a cada lote novo, calcula o PSI (Population Stability Index) de cada
atributo entre a referência (dados já conhecidos) e o lote que chegou.
  PSI < 0,10        estável
  0,10 a 0,25       atenção
  >= 0,25           ALERTA: mudança relevante de distribuição
O resultado vai para o MLflow, para o manifesto do modelo e, no GitHub Actions,
vira uma anotação de aviso e uma tabela no resumo da execução.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

EPS = 1e-4


def _psi(p_ref: np.ndarray, p_novo: np.ndarray) -> float:
    p_ref = np.clip(p_ref, EPS, None)
    p_novo = np.clip(p_novo, EPS, None)
    return float(np.sum((p_novo - p_ref) * np.log(p_novo / p_ref)))


def psi_numerico(ref: pd.Series, novo: pd.Series, bins: int = 10) -> float:
    ref, novo = ref.dropna(), novo.dropna()
    if ref.empty or novo.empty:
        return 0.0 if ref.empty and novo.empty else 1.0  # coluna "apareceu" ou "sumiu"
    cortes = np.unique(np.quantile(ref, np.linspace(0, 1, bins + 1)))
    if len(cortes) < 2:  # referência constante
        return _psi(np.array([1.0, 0.0]), np.array([(novo == ref.iloc[0]).mean(), (novo != ref.iloc[0]).mean()]))
    cortes[0], cortes[-1] = -np.inf, np.inf
    p_ref = np.histogram(ref, cortes)[0] / len(ref)
    p_novo = np.histogram(novo, cortes)[0] / len(novo)
    return _psi(p_ref, p_novo)


def psi_categorico(ref: pd.Series, novo: pd.Series) -> float:
    cats = sorted(set(ref.dropna().astype(str)) | set(novo.dropna().astype(str)))
    p_ref = ref.astype(str).value_counts(normalize=True).reindex(cats, fill_value=0).values
    p_novo = novo.astype(str).value_counts(normalize=True).reindex(cats, fill_value=0).values
    return _psi(p_ref, p_novo)


def relatorio_drift(ref: pd.DataFrame, novo: pd.DataFrame, cfg: dict) -> pd.DataFrame:
    d = cfg["drift"]
    linhas = []
    for col in ref.columns:
        if pd.api.types.is_numeric_dtype(ref[col]):
            psi = psi_numerico(ref[col], novo[col], d["bins"])
        else:
            psi = psi_categorico(ref[col], novo[col])
        status = "ALERTA" if psi >= d["psi_alerta"] else ("ATENÇÃO" if psi >= d["psi_atencao"] else "estável")
        linhas.append({"feature": col, "psi": round(psi, 4), "status": status})
    return pd.DataFrame(linhas).sort_values("psi", ascending=False).reset_index(drop=True)
