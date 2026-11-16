# Projeto N3 · Pipeline MLOps · Bank Marketing

Trabalho N3 de Análise Preditiva (Prof. Rodrigo Hagstrom) · Católica de Santa Catarina

**Integrantes:** Miguel, Julia, Ana Olska, Ana Paula e Eduardo

## Resultado de referência (teste local, 9 ciclos)

| Ciclo | Lote | Registros novos | AUC-PR candidato | AUC-PR vigente anterior | Decisão |
|---|---|---|---|---|---|
| 1 | mai–nov/2008 | 27.680 | 0,104 | — | PROMOVIDO (RF) |
| 2 | dez/08–abr/09 | 2.750 | 0,204 | 0,107 | PROMOVIDO |
| 3 | mai–jul/09 | 6.687 | 0,256 | 0,222 | PROMOVIDO |
| 4 | ago–out/09 | 1.484 | 0,321 | 0,282 | PROMOVIDO |
| 5 | nov/09–mar/10 | 793 | 0,363 | 0,347 | PROMOVIDO |
| 6 | abr–jun/10 | 615 | 0,415 | 0,390 | PROMOVIDO |
| 7 | jul–set/10 | 847 | 0,458 | 0,451 | PROMOVIDO |
| 8 | out–nov/10 | 332 | 0,463 | 0,467 | MANTIDO |
| 9 | (sem lote novo) | 0 | 0,463 | 0,467 | MANTIDO |

Prevalência da classe positiva ≈ 11%, então um modelo aleatório teria AUC-PR ≈ 0,11. No último ciclo, a AUC-ROC do vigente é ≈ 0,81.

Pipeline MLOps agendado que, a cada ciclo, incorpora um novo lote de dados, elege as features por votação de três métodos, coloca dois algoritmos para competir, registra tudo no MLflow e decide automaticamente se o modelo vigente deve ser substituído.

## Problema

Prever se um cliente vai contratar um **depósito a prazo** após uma ligação de telemarketing de um banco português, para que o banco ligue primeiro para quem tem mais chance de aderir.

| Item | Definição |
|---|---|
| Dados | [Bank Marketing (UCI)](https://archive.ics.uci.edu/dataset/222/bank+marketing), arquivo `bank-additional-full.csv`, licença CC BY 4.0 |
| Unidade de observação | Um contato telefônico com um cliente (41.188 contatos, mai/2008 a nov/2010) |
| Alvo | `y`: contratou o depósito (1) ou não (0). Classe positiva ≈ 11% |
| Métrica principal | AUC-PR (`average_precision`), maximizar |
| Particionamento | 70/15/15 (treino/validação/teste) estável por hash do `registro_id` |
| Chegada de dados | Lote inicial = mai a nov/2008. Depois, 1 lote por ciclo com os 3 meses seguintes de campanha (8 lotes no total) |

## Fluxo de cada ciclo

```
schedule (cron) ──> 01_etl ──> 02_votacao_features ──> 03_treino_competicao ──> 04_selecao_promocao ──> 05_previsoes
                     │             │                        │                         │                      │
                     │ ingere lote │ 3 métodos + votação    │ Modelo A (Reg. Log.)    │ vencedor do ciclo    │ previsões para a
                     │ ETL + drift │ (>= 2 de 3 votos)      │ Modelo B (Random Forest)│ × vigente → promove  │ próxima campanha
                     │ cycle_id    │                        │ mesmas features e CV    │ ou mantém            │ + comparação de ciclos
                     └─────────────┴──────── todos registram no MLflow com o mesmo cycle_id e data_version ─────┘
```

## Estrutura

```
.github/workflows/pipeline_mlops.yml   workflow único (schedule + workflow_dispatch)
config/config.yaml                     dados, métrica, votação, algoritmos, regra de promoção, drift
data/source/                           base completa da UCI (simula o sistema de origem)
data/raw/                              lotes já incorporados (lote_000.csv, lote_001.csv, ...)
data/state/lotes_ingeridos.json        registro de ingestão (hash de cada lote)
notebooks/01..05                       documentação do processo; chamam funções de src/
src/                                   ingestao, preprocessamento, features, modelagem, promocao, drift, utils
models/modelo_vigente.json             manifesto do modelo vigente
models/vigente/modelo.joblib           pipeline vigente (features + pré-processamento + modelo)
models/historico/                      todos os modelos já promovidos (nada é apagado)
models/historico_decisoes.csv          decisão de cada ciclo
mlflow.db + mlartifacts/               tracking do MLflow (histórico de todos os ciclos)
evidencias/<cycle_id>/                 notebooks executados, votação, CV, manifesto e previsões do ciclo
executar_pipeline.sh                   roda um ciclo completo (usado pelo workflow e localmente)
```

## Como executar localmente

```bash
python -m venv .venv && source .venv/bin/activate      # Windows: .venv\Scripts\activate
pip install -r requirements.txt
bash executar_pipeline.sh                              # roda 1 ciclo (01 -> 05)
```

Cada execução incorpora o próximo lote. Rodar de novo simula o ciclo seguinte. Depois do 8º lote, os ciclos continuam rodando com a mesma `data_version`, e o modelo vigente tende a ser mantido.

Para **zerar** e começar do ciclo 1:

```bash
rm -rf data/raw/* data/state/* models/* mlflow.db mlartifacts evidencias/*
```

## Execução agendada (GitHub Actions)

1. Suba o repositório para o GitHub (público ou privado com acesso para o professor).
2. Em **Settings → Actions → General → Workflow permissions**, marque **Read and write permissions** (o workflow salva o estado do ciclo no próprio repositório).
3. O workflow `pipeline-mlops` roda **todo dia às 08:17 (Brasília)**. Para depurar, use **Actions → pipeline-mlops → Run workflow**. Execução manual **não** substitui a agendada.
4. Em cada execução:
   - o checkout traz o estado persistente do ciclo anterior (lotes, manifesto, modelos, MLflow);
   - `executar_pipeline.sh` roda 01 → 05 em sequência; qualquer erro interrompe o job;
   - só se tudo der certo, o estado novo é commitado (`ciclo cXXX (schedule) [skip ci]`). Uma falha nunca promove modelo;
   - o pacote do ciclo (MLflow, modelos, evidências) é publicado como **artifact** baixável;
   - o resumo da execução mostra o lote, o drift e a decisão de promoção.

> O GitHub pode atrasar agendamentos em até ~1 h e desativa schedules de repositórios sem atividade por 60 dias.
>
> São 8 lotes, então em 8 dias agendados a fonte se esgota. Depois disso os ciclos seguem rodando (mesma `data_version`, vigente mantido). Desative o workflow após a entrega para o repositório não crescer à toa.

**Prova do gatilho:** o campo `gatilho` (`schedule`, `workflow_dispatch` ou `local`) é gravado no contexto do ciclo, nos runs do MLflow, no manifesto e em `models/historico_decisoes.csv`.

## Consultar o MLflow

```bash
mlflow ui --backend-store-uri sqlite:///mlflow.db
# abrir http://127.0.0.1:5000 → experimento "bank-marketing-n3"
```

Cada ciclo gera 6 runs com as tags `cycle_id`, `data_version` e `etapa`: `01_etl`, `02_votacao`, `03_modelo_A_*`, `03_modelo_B_*`, `04_decisao` e `05_previsoes`. Filtro útil: `tags.cycle_id = "c002-..."`.

Os caminhos dos artefatos são reescritos automaticamente para a pasta local na primeira vez que qualquer notebook roda (`src/utils.py`). Assim, o histórico gerado no GitHub abre normalmente no seu computador.

## Decisões de projeto

**ETL sem vazamento.** Limpeza determinística (duplicatas, remoção de `duration`, criação de atributos) antes do split. Tudo que aprende com os dados (imputação, winsorização por percentil, padronização, one-hot) fica em um `Pipeline` ajustado só no treino.

**Votação de features.** SelectKBest (informação mútua), importância por Random Forest e RFE com Regressão Logística. Cada método aprova seu top 8, e a feature é eleita com ≥ 2 votos (mínimo de 6, completando pela menor soma de rankings). Na validação cruzada, a votação é refeita dentro de cada fold.

**Promoção.** O vencedor do ciclo (maior AUC-PR média na CV) enfrenta o vigente na mesma validação. Ele só é promovido se superar o vigente em pelo menos 0,005 e for melhor que um modelo aleatório. O teste fica reservado para a avaliação final.

**Por que hash e não corte temporal no teste?** Testamos o corte temporal. Como o volume se concentra em 2008-2009 e a campanha muda de perfil depois, o trecho mais recente sempre caía em um regime nunca visto, com AUC ≈ 0,5 em todos os ciclos. Cada contato é independente. Por isso, a ordem do tempo é respeitada na chegada dos lotes e vigiada pelo monitor de drift.

## Recurso adicional (Critério 10): monitoramento de drift com alerta

| | |
|---|---|
| Problema | O perfil dos dados muda muito entre 2008 e 2010 (crise financeira, queda da Euribor, mudança no público da campanha). Um modelo pode envelhecer sem ninguém perceber. |
| Solução | A cada lote novo, o notebook 01 calcula o PSI de cada atributo (lote novo × dados já conhecidos). PSI ≥ 0,25 gera alerta. |
| Integração | Alerta como anotação no GitHub Actions + tabela no resumo da execução; PSI registrado no MLflow; flag `drift_alerta` gravada no manifesto do modelo e no histórico de decisões. |
| Evidência | Gráfico de PSI no notebook 01 executado de cada ciclo (`evidencias/<cycle_id>/`) e coluna `drift_alerta` em `models/historico_decisoes.csv`. |
