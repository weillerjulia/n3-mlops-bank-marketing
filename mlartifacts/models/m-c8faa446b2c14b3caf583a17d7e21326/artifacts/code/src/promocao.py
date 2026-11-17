"""Eleição do vencedor do ciclo e promoção controlada do modelo vigente."""
from __future__ import annotations

import shutil
from datetime import datetime, timezone

import joblib
import pandas as pd

from .utils import caminho, ler_json, salvar_json

MANIFESTO = "models/modelo_vigente.json"
VIGENTE = "models/vigente/modelo.joblib"
HISTORICO_DIR = "models/historico"
HISTORICO_DECISOES = "models/historico_decisoes.csv"


def melhor_que(a: float, b: float, direcao: str, margem: float = 0.0) -> bool:
    return a >= b + margem if direcao == "maximize" else a <= b - margem


def eleger_vencedor(cv: pd.DataFrame, metrica: str, direcao: str) -> tuple[str, pd.DataFrame]:
    """Compara a média da métrica principal na validação cruzada (mesmos folds para A e B).
    Empate exato: vence o modelo com menor desvio-padrão (mais estável)."""
    resumo = (cv.groupby(["modelo", "algoritmo"])[metrica]
                .agg(["mean", "std"]).reset_index()
                .rename(columns={"mean": f"{metrica}_media", "std": f"{metrica}_desvio"}))
    asc = direcao != "maximize"
    resumo = resumo.sort_values([f"{metrica}_media", f"{metrica}_desvio"], ascending=[asc, True])
    return resumo.iloc[0]["modelo"], resumo.reset_index(drop=True)


def decidir_promocao(score_candidato: float, score_vigente: float | None, piso: float | None,
                     cfg: dict) -> dict:
    p = cfg["promocao"]
    direcao = cfg["avaliacao"]["direcao"]
    limiar = p["limiar_melhoria"]
    if piso is not None and p.get("exigir_acima_do_aleatorio") and not melhor_que(score_candidato, piso, direcao):
        return {"decisao": "REJEITADO",
                "motivo": f"Candidato ({score_candidato:.4f}) não supera o modelo aleatório ({piso:.4f})."}
    if score_vigente is None:
        return {"decisao": "PROMOVIDO", "motivo": "Não havia modelo vigente (primeiro ciclo)."}
    if melhor_que(score_candidato, score_vigente, direcao, limiar):
        return {"decisao": "PROMOVIDO",
                "motivo": f"Candidato ({score_candidato:.4f}) supera o vigente ({score_vigente:.4f}) "
                          f"em pelo menos {limiar} na validação do ciclo."}
    return {"decisao": "MANTIDO",
            "motivo": f"Candidato ({score_candidato:.4f}) não supera o vigente ({score_vigente:.4f}) "
                      f"pelo limiar de {limiar}; vigente preservado."}


def ler_manifesto() -> dict | None:
    return ler_json(caminho(MANIFESTO))


def carregar_vigente():
    m = ler_manifesto()
    if m is None:
        return None, None
    arq = caminho(m["arquivo_modelo"])
    if not arq.exists():
        raise FileNotFoundError(f"Manifesto aponta para {arq}, que não existe no armazenamento persistente.")
    return joblib.load(arq), m


def promover(pipeline, manifesto: dict) -> dict:
    """Serializa o novo vigente, preservando todos os anteriores em models/historico."""
    hist = caminho(HISTORICO_DIR)
    hist.mkdir(parents=True, exist_ok=True)
    arq_hist = hist / f"{manifesto['cycle_id']}_{manifesto['algoritmo']}.joblib"
    joblib.dump(pipeline, arq_hist, compress=3)
    caminho(VIGENTE).parent.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(arq_hist, caminho(VIGENTE))

    anterior = ler_manifesto()
    manifesto = {**manifesto,
                 "arquivo_modelo": str(arq_hist.relative_to(caminho("."))),
                 "promovido_em": datetime.now(timezone.utc).isoformat(timespec="seconds"),
                 "substituiu": (anterior or {}).get("cycle_id")}
    salvar_json(manifesto, caminho(MANIFESTO))
    return manifesto


def registrar_decisao(linha: dict) -> pd.DataFrame:
    """Histórico de decisões de todos os ciclos (auditoria entre ciclos)."""
    arq = caminho(HISTORICO_DECISOES)
    df = pd.read_csv(arq) if arq.exists() else pd.DataFrame()
    df = pd.concat([df, pd.DataFrame([linha])], ignore_index=True)
    df.to_csv(arq, index=False)
    return df
