"""Utilidades comuns: caminhos, configuração, contexto do ciclo e MLflow."""
from __future__ import annotations

import json
import os
import sqlite3
import subprocess
from datetime import datetime, timezone
from pathlib import Path

import yaml


def raiz_projeto() -> Path:
    """Encontra a raiz do repositório (pasta que contém config/config.yaml)."""
    p = Path.cwd().resolve()
    for cand in [p, *p.parents]:
        if (cand / "config" / "config.yaml").exists():
            return cand
    raise FileNotFoundError("Não encontrei config/config.yaml acima de " + str(p))


ROOT = raiz_projeto()


def caminho(rel: str | Path) -> Path:
    return ROOT / rel


def carregar_config() -> dict:
    with open(caminho("config/config.yaml"), encoding="utf-8") as f:
        return yaml.safe_load(f)


def ler_json(path: Path, padrao=None):
    if not Path(path).exists():
        return padrao
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def salvar_json(obj, path: Path) -> None:
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, ensure_ascii=False, indent=2, default=str)


# ---------------------------------------------------------------- contexto do ciclo
def caminho_contexto(cfg: dict) -> Path:
    return caminho(cfg["dados"]["state_dir"]) / "ciclo_atual.json"


def ler_contexto(cfg: dict) -> dict:
    ctx = ler_json(caminho_contexto(cfg))
    if ctx is None:
        raise RuntimeError("Contexto do ciclo não encontrado: execute o notebook 01 primeiro.")
    return ctx


def atualizar_contexto(cfg: dict, **campos) -> dict:
    ctx = ler_json(caminho_contexto(cfg), {})
    ctx.update(campos)
    salvar_json(ctx, caminho_contexto(cfg))
    return ctx


def info_execucao() -> dict:
    """Metadados de quem disparou a execução (prova do gatilho schedule)."""
    commit = None
    try:
        commit = subprocess.check_output(["git", "rev-parse", "--short", "HEAD"], cwd=ROOT,
                                         stderr=subprocess.DEVNULL).decode().strip()
    except Exception:
        pass
    return {
        "gatilho": os.environ.get("GITHUB_EVENT_NAME", "local"),
        "github_run_id": os.environ.get("GITHUB_RUN_ID"),
        "github_run_url": (f"{os.environ['GITHUB_SERVER_URL']}/{os.environ['GITHUB_REPOSITORY']}"
                           f"/actions/runs/{os.environ['GITHUB_RUN_ID']}")
        if os.environ.get("GITHUB_RUN_ID") else None,
        "commit": commit,
        "iniciado_em": datetime.now(timezone.utc).isoformat(timespec="seconds"),
    }


# ---------------------------------------------------------------- MLflow
def _normalizar_caminhos_mlflow(db_path: Path, artifact_root: Path) -> None:
    """O MLflow grava caminhos absolutos dos artefatos. Como o estado é versionado
    no Git e roda em máquinas diferentes (runner do GitHub, PC do aluno), reescreve
    o prefixo até '/mlartifacts' para o caminho local atual."""
    if not db_path.exists():
        return
    alvo = str(artifact_root)
    colunas = [("experiments", "artifact_location"), ("runs", "artifact_uri"),
               ("logged_models", "artifact_location"), ("model_versions", "storage_location")]
    con = sqlite3.connect(db_path)
    try:
        for tabela, col in colunas:
            try:
                linhas = con.execute(f"SELECT rowid, {col} FROM {tabela}").fetchall()
            except sqlite3.OperationalError:
                continue
            for rowid, valor in linhas:
                if not valor or "/mlartifacts" not in valor:
                    continue
                prefixo = "file://" if valor.startswith("file://") else ""
                resto = valor.split("/mlartifacts", 1)[1]
                novo = f"{prefixo}{alvo}{resto}"
                if novo != valor:
                    con.execute(f"UPDATE {tabela} SET {col}=? WHERE rowid=?", (novo, rowid))
        con.commit()
    finally:
        con.close()


def configurar_mlflow(cfg: dict):
    os.environ.setdefault("MLFLOW_DISABLE_AGENT_HINT", "1")
    import mlflow

    db_rel = cfg["mlflow"]["tracking_uri"].replace("sqlite:///", "")
    db_path = caminho(db_rel)
    artifact_root = caminho(cfg["mlflow"]["artifact_dir"])
    _normalizar_caminhos_mlflow(db_path, artifact_root)

    mlflow.set_tracking_uri(f"sqlite:///{db_path}")
    nome = cfg["mlflow"]["experimento"]
    exp = mlflow.get_experiment_by_name(nome)
    if exp is None:
        mlflow.create_experiment(nome, artifact_location=str(artifact_root))
    mlflow.set_experiment(nome)
    return mlflow


def tags_ciclo(ctx: dict, etapa: str) -> dict:
    return {
        "cycle_id": ctx["cycle_id"],
        "data_version": ctx["data_version"],
        "etapa": etapa,
        "gatilho": ctx.get("execucao", {}).get("gatilho", "local"),
    }


def resumo_github(texto_md: str) -> None:
    """Escreve no resumo da execução do GitHub Actions (quando disponível)."""
    destino = os.environ.get("GITHUB_STEP_SUMMARY")
    if destino:
        with open(destino, "a", encoding="utf-8") as f:
            f.write(texto_md + "\n")


def aviso_github(msg: str) -> None:
    """Gera uma anotação de aviso visível na execução do GitHub Actions."""
    linha = "::warning title=Drift detectado::" + msg.replace("%", "%25").replace("\n", "%0A")
    # O nbconvert guarda os prints dentro do .ipynb, onde o GitHub não os lê. Por isso o
    # aviso vai para um arquivo, que o executar_pipeline.sh repete no log do job.
    destino = os.environ.get("AVISOS_PIPELINE")
    if destino:
        with open(caminho(destino), "a", encoding="utf-8") as f:
            f.write(linha + "\n")
