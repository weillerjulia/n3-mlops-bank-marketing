"""Entrada e versionamento de dados novos.

O arquivo-fonte da UCI simula o sistema de origem do banco. Os registros estão em
ordem cronológica (mai/2008 a nov/2010), mas sem coluna de ano. Este módulo
reconstrói o período (AAAA-MM) de cada registro a partir das trocas de mês, e
divide a base em lotes que chegam um por ciclo.
"""
from __future__ import annotations

import hashlib
from datetime import datetime, timezone

import pandas as pd

from .utils import caminho, ler_json, salvar_json

MESES = {"jan": 1, "feb": 2, "mar": 3, "apr": 4, "may": 5, "jun": 6,
         "jul": 7, "aug": 8, "sep": 9, "oct": 10, "nov": 11, "dec": 12}


def carregar_fonte(cfg: dict) -> pd.DataFrame:
    """Lê a fonte e adiciona registro_id (posição original) e periodo (AAAA-MM)."""
    d = cfg["dados"]
    df = pd.read_csv(caminho(d["fonte"]), sep=d["separador"])
    df.insert(0, "registro_id", range(len(df)))

    ano = int(d["periodo_inicio"][:4])
    mes_anterior = None
    periodos = []
    for m in df["month"].map(MESES):
        if mes_anterior is not None and m < mes_anterior:
            ano += 1  # virada de ano: o mês "voltou"
        periodos.append(f"{ano}-{m:02d}")
        mes_anterior = m
    df["periodo"] = periodos

    if df["periodo"].iloc[-1] != d["periodo_fim"]:
        raise ValueError(f"Reconstrução de períodos inconsistente: termina em {df['periodo'].iloc[-1]}")
    return df


def plano_de_lotes(cfg: dict, fonte: pd.DataFrame) -> list[list[str]]:
    """Lote 0 = períodos até 'lote_inicial_ate'; depois, grupos de N períodos."""
    d = cfg["dados"]
    periodos = list(dict.fromkeys(fonte["periodo"]))  # ordem cronológica, sem repetição
    inicial = [p for p in periodos if p <= d["lote_inicial_ate"]]
    resto = [p for p in periodos if p > d["lote_inicial_ate"]]
    n = d["periodos_por_lote"]
    return [inicial] + [resto[i:i + n] for i in range(0, len(resto), n)]


def _caminho_registro(cfg):
    return caminho(cfg["dados"]["state_dir"]) / "lotes_ingeridos.json"


def lotes_ingeridos(cfg: dict) -> list[dict]:
    return ler_json(_caminho_registro(cfg), [])


def ingerir_proximo_lote(cfg: dict) -> dict:
    """Copia o próximo lote da fonte para data/raw e registra a ingestão.

    Retorna um dicionário descrevendo o lote ou {'novo': False} se a fonte não
    tiver mais dados (o ciclo roda mesmo assim, com a mesma data_version)."""
    fonte = carregar_fonte(cfg)
    plano = plano_de_lotes(cfg, fonte)
    registro = lotes_ingeridos(cfg)
    idx = len(registro)
    if idx >= len(plano):
        return {"novo": False, "lote": None, "periodos": [], "linhas": 0}

    periodos = plano[idx]
    lote = fonte[fonte["periodo"].isin(periodos)]

    # Incorporação sem duplicação: ignora registro_id já ingerido
    ja = set(carregar_acumulado(cfg)["registro_id"]) if registro else set()
    lote = lote[~lote["registro_id"].isin(ja)]

    raw = caminho(cfg["dados"]["raw_dir"])
    raw.mkdir(parents=True, exist_ok=True)
    arquivo = raw / f"lote_{idx:03d}.csv"
    lote.to_csv(arquivo, index=False)

    info = {
        "lote": idx,
        "arquivo": str(arquivo.relative_to(caminho("."))),
        "periodos": periodos,
        "linhas": int(len(lote)),
        "sha256": hashlib.sha256(arquivo.read_bytes()).hexdigest()[:12],
        "ingerido_em": datetime.now(timezone.utc).isoformat(timespec="seconds"),
    }
    registro.append(info)
    salvar_json(registro, _caminho_registro(cfg))
    return {"novo": True, **info}


def carregar_acumulado(cfg: dict) -> pd.DataFrame:
    """Concatena todos os lotes já ingeridos, em ordem cronológica e sem duplicar."""
    registro = lotes_ingeridos(cfg)
    if not registro:
        raise RuntimeError("Nenhum lote ingerido ainda.")
    partes = [pd.read_csv(caminho(r["arquivo"])) for r in registro]
    df = pd.concat(partes, ignore_index=True)
    df = df.drop_duplicates(subset="registro_id").sort_values("registro_id").reset_index(drop=True)
    return df


def data_version(cfg: dict) -> str:
    """Versão dos dados = nº de lotes + hash dos hashes dos arquivos."""
    registro = lotes_ingeridos(cfg)
    h = hashlib.sha256("".join(r["sha256"] for r in registro).encode()).hexdigest()[:8]
    return f"v{len(registro):03d}-{h}"


def lote_pendente(cfg: dict) -> pd.DataFrame | None:
    """Próximo lote ainda NÃO ingerido (clientes da próxima campanha).
    Usado no notebook 05 para previsões em registros nunca vistos."""
    fonte = carregar_fonte(cfg)
    plano = plano_de_lotes(cfg, fonte)
    idx = len(lotes_ingeridos(cfg))
    if idx >= len(plano):
        return None
    return fonte[fonte["periodo"].isin(plano[idx])].reset_index(drop=True)
