# Projeto N3 · Pipeline MLOps · Bank Marketing

Trabalho N3 de Análise Preditiva (Prof. Rodrigo Hagstrom) · Católica de Santa Catarina

**Integrantes:** Miguel, Julia, Ana Olska, Ana Paula e Eduardo

Pipeline MLOps agendado que, a cada ciclo, incorpora um novo lote de dados, elege as features por votação de três métodos, coloca dois algoritmos para competir, registra tudo no MLflow e decide automaticamente se o modelo vigente deve ser substituído.

## Resultados agendados (até o c007, 22/11/2026)

Histórico completo e atual: [`models/historico_decisoes.csv`](models/historico_decisoes.csv).

| Ciclo | Data | AUC-PR candidato (val.) | AUC-PR vigente anterior (val.) | Decisão | AUC-ROC teste | AUC-ROC campanha seguinte |
|---|---|---|---|---|---|---|
| c001 | 16/11 | 0,104 | — | PROMOVIDO | 0,571 | 0,597 |
| c002 | 17/11 | 0,204 | 0,107 | PROMOVIDO | 0,655 | 0,618 |
| c003 | 18/11 | 0,256 | 0,222 | PROMOVIDO | 0,687 | 0,543 |
| c004 | 19/11 | 0,321 | 0,282 | PROMOVIDO | 0,717 | 0,539 |
| c005 | 20/11 | 0,363 | 0,347 | PROMOVIDO | 0,750 | 0,647 |
| c006 | 21/11 | 0,415 | 0,390 | PROMOVIDO | 0,765 | 0,755 |
| c007 | 22/11 | 0,458 | 0,451 | PROMOVIDO | 0,783 | 0,727 |

O teste por hash mede desempenho em períodos já conhecidos e é otimista para o futuro. A última coluna é o desempenho fora do tempo.

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

No teste local, o vigente do último ciclo tem AUC-ROC ≈ 0,82 na validação e ≈ 0,79 no teste.

## Problema

Prever se um cliente vai contratar um **depósito a prazo** após uma ligação de telemarketing de um banco português, para que o banco ligue primeiro para quem tem mais chance de aderir.

| Item | Definição |
|---|---|
| Dados | [Bank Marketing (UCI)](https://archive.ics.uci.edu/dataset/222/bank+marketing), arquivo `bank-additional-full.csv`, licença CC BY 4.0. Moro, S., Cortez, P., Rita, P. (2014), *A data-driven approach to predict the success of bank telemarketing*, Decision Support Systems, 62, 22-31 |
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
reports/relatorio_n3.pdf               relatório técnico (gerado a partir dos artefatos)
reports/gerar_relatorio.py             gera o relatório em PDF
```

## Relatório técnico

O relatório da entrega está em [`reports/relatorio_n3.pdf`](reports/relatorio_n3.pdf). O script `reports/gerar_relatorio.py` monta o PDF e tira os resultados das evidências de cada ciclo, o histórico de decisões, o manifesto do modelo vigente, o `config.yaml` e o `mlflow.db`. As descrições de método, o teste local de referência e a verificação do corte temporal são texto fixo no script e precisam de revisão se o código mudar.

Para atualizar o relatório depois de novos ciclos agendados:

```bash
git pull
python reports/gerar_relatorio.py
```

Com dois ciclos ou mais, o relatório inclui sozinho as mudanças de features entre ciclos, o gráfico do vencedor de cada ciclo × vigente e a tabela de drift.

## Como executar localmente

```bash
python -m venv .venv && source .venv/bin/activate      # Windows: .venv\Scripts\activate
pip install -r requirements.txt
bash executar_pipeline.sh                              # roda 1 ciclo (01 -> 05)
```

> Um ciclo local altera lotes, modelos e `mlflow.db`. Rode numa cópia do repositório e não commite esse estado, para não misturar com o histórico agendado.

Cada execução incorpora o próximo lote. Rodar de novo simula o ciclo seguinte. Depois do 8º lote, os ciclos continuam rodando com a mesma `data_version`, e o modelo vigente tende a ser mantido.

Para **zerar** e começar do ciclo 1 (só numa cópia, pois apaga o histórico):

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
   - o resumo da execução mostra o lote, o drift e a decisão de promoção; se houver drift, sai também um aviso nas anotações do job.

> O GitHub pode atrasar bastante os agendamentos (de 6,7 a 9,8 h entre c001 e c007) e desativa schedules de repositórios sem atividade por 60 dias.
>
> São 8 lotes, então em 8 dias agendados a fonte se esgota. Depois disso os ciclos seguem rodando (mesma `data_version`, vigente mantido). Desative o workflow após a entrega para o repositório não crescer à toa.

**Prova do gatilho:** o campo `gatilho` (`schedule`, `workflow_dispatch` ou `local`) é gravado no contexto do ciclo, nos runs do MLflow, no manifesto e em `models/historico_decisoes.csv`.

## Consultar o MLflow

O banco guarda os caminhos do runner do GitHub. Numa cópia do repositório, reescreva os caminhos antes de abrir a interface (sem isso, os artefatos aparecem vazios):

```bash
python -c "from src.utils import carregar_config, configurar_mlflow; configurar_mlflow(carregar_config())"
mlflow ui --backend-store-uri sqlite:///mlflow.db
# abrir http://127.0.0.1:5000 → experimento "bank-marketing-n3"
```

A reescrita altera o `mlflow.db`: não commite esse arquivo.

Cada ciclo gera 6 runs com as tags `cycle_id`, `data_version`, `gatilho` e `etapa` (`01_etl`, `02_votacao`, `03_treino` duas vezes, com a tag `modelo` = A ou B, `04_promocao` e `05_previsoes`). Filtro útil: `tags.cycle_id = "c007-20261122-1830"`.

## Reproduzir as previsões

Na raiz do repositório, com as dependências instaladas (no Windows, use o Git Bash). O comando refaz as previsões do último ciclo com o modelo vigente e compara com o CSV salvo, sem alterar nada:

```bash
python - <<'EOF'
import glob
import pandas as pd
from src.utils import carregar_config
from src.ingestao import lote_pendente, carregar_acumulado
from src.preprocessamento import limpar_e_criar_atributos, particionar, separar_xy
from src.promocao import carregar_vigente

cfg = carregar_config()
modelo, manifesto = carregar_vigente()
pendente = lote_pendente(cfg)
if pendente is not None:  # próxima campanha
    dados = limpar_e_criar_atributos(pendente, cfg)
else:  # fonte esgotada: o notebook 05 usa o teste reservado
    dados = particionar(limpar_e_criar_atributos(carregar_acumulado(cfg), cfg), cfg)[2]
dados["p"] = modelo.predict_proba(separar_xy(dados, cfg)[0])[:, 1]
salvo = pd.read_csv(sorted(glob.glob("evidencias/c*/previsoes.csv"))[-1])
junto = salvo.merge(dados[["registro_id", "p"]], on="registro_id")
assert len(junto) == len(salvo) == len(dados), "registros diferentes"
assert (junto.p - junto.probabilidade_adesao).abs().max() < 1e-3, "probabilidades diferentes"
print(f"{len(junto)} previsões reproduzidas com o modelo {manifesto['cycle_id']}")
EOF
```

## Decisões de projeto

**ETL sem vazamento.** Limpeza determinística (duplicatas, remoção de `duration`, criação de atributos) antes do split. Tudo que aprende com os dados (imputação, winsorização por percentil, padronização, one-hot) fica em um `Pipeline` ajustado só no treino.

**Votação de features.** SelectKBest (informação mútua), importância por Random Forest e RFE com Regressão Logística. Cada método aprova seu top 8 (empates na 8ª posição também recebem voto; a informação mútua usa `mutual_info_classif`, o mesmo critério do SelectKBest, agregado por variável), e a feature é eleita com ≥ 2 votos (mínimo de 6, completando pela menor soma de rankings). Na validação cruzada, a votação é refeita dentro de cada fold.

**Promoção.** O vencedor do ciclo (maior AUC-PR média na CV) enfrenta o vigente na mesma validação. Ele só é promovido se superar o vigente em pelo menos 0,005 e for melhor que um modelo aleatório. O teste fica reservado para a avaliação final.

**Por que hash e não corte temporal no teste?** O hash mantém as partições estáveis, o que permite comparar vigente e candidato sem vazamento entre ciclos. Num corte temporal 70/15/15, a AUC-ROC no trecho mais recente variou de 0,41 a 0,69. O custo do hash é um teste otimista: contatos da mesma época têm os mesmos indicadores macroeconômicos, e quase todas as linhas de teste (99% no c006) têm atributos eleitos idênticos a alguma linha de treino. Por isso o relatório mostra também o desempenho na campanha seguinte.

## Recurso adicional (Critério 10): monitoramento de drift com alerta

| | |
|---|---|
| Problema | O perfil dos dados muda muito entre 2008 e 2010 (crise financeira, queda da Euribor, mudança no público da campanha). Um modelo pode envelhecer sem ninguém perceber. |
| Solução | A cada lote novo, o notebook 01 calcula o PSI de cada atributo (lote novo × dados já conhecidos). PSI ≥ 0,25 gera alerta. |
| Integração | Alerta como anotação no GitHub Actions + linha no resumo da execução; PSI registrado no MLflow; flag `drift_alerta` gravada no manifesto do modelo e no histórico de decisões. |
| Evidência | Gráfico de PSI no notebook 01 executado de cada ciclo, `avisos.txt` (ciclos com alerta após a correção), ambos em `evidencias/<cycle_id>/`, e coluna `drift_alerta` em `models/historico_decisoes.csv`. |

**Correção em 22/11/2026 (`f17d0df`):** de c002 a c006 o aviso não chegava às anotações do GitHub, e o PSI de `contatado_antes` (c002–c006) e de `previous` (c002–c004) saía 0 por erro de binning. As evidências originais foram preservadas, e o relatório traz os valores recalculados. O c007, primeiro ciclo agendado após a correção, gerou a anotação "Drift detectado" no [run 45055360879](https://github.com/weillerjulia/n3-mlops-bank-marketing/actions/runs/45055360879), com 14 atributos, entre eles `contatado_antes` (PSI 0,967) e `previous` (1,089).
