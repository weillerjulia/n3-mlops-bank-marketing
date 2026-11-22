#!/usr/bin/env bash
# Executa um ciclo completo do pipeline (01 -> 05), na ordem, parando no primeiro erro.
# Usado tanto pelo GitHub Actions quanto localmente.
set -euo pipefail
cd "$(dirname "$0")"

TMP=evidencias/_ciclo_em_execucao
rm -rf "$TMP"; mkdir -p "$TMP"
export AVISOS_PIPELINE="$TMP/avisos.txt"

executar () {
  echo "==> $1"
  jupyter nbconvert --to notebook --execute "notebooks/$1.ipynb" \
    --ExecutePreprocessor.timeout=1800 --output-dir "$TMP" --output "$1_executado.ipynb"
}

executar 01_etl
# Comandos ::warning:: só viram anotação se saírem no log do job (fora do notebook)
if [ -f "$AVISOS_PIPELINE" ]; then cat "$AVISOS_PIPELINE"; fi
CYCLE_ID=$(python -c "import json;print(json.load(open('data/state/ciclo_atual.json'))['cycle_id'])")
executar 02_votacao_features
executar 03_treino_competicao
executar 04_selecao_promocao
executar 05_previsoes

# Evidências do ciclo: notebooks executados + contexto + votação + CV
DEST="evidencias/$CYCLE_ID"
mkdir -p "$DEST"
mv "$TMP"/*.ipynb "$DEST"/
if [ -f "$AVISOS_PIPELINE" ]; then mv "$AVISOS_PIPELINE" "$DEST"/avisos.txt; fi
cp data/state/ciclo_atual.json data/state/votacao_ciclo.json data/state/cv_ciclo.csv \
   data/state/rankings_features.png "$DEST"/
cp models/modelo_vigente.json "$DEST"/modelo_vigente_apos_ciclo.json
rm -rf "$TMP"
echo "CYCLE_ID=$CYCLE_ID"
[ -n "${GITHUB_OUTPUT:-}" ] && echo "cycle_id=$CYCLE_ID" >> "$GITHUB_OUTPUT"
echo "Ciclo $CYCLE_ID concluído."
