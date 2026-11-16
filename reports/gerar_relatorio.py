"""Gera reports/relatorio_n3.pdf a partir dos artefatos versionados no repositório.

Tudo que é número no relatório vem de arquivos produzidos pelo pipeline:
  - evidencias/<cycle_id>/ciclo_atual.json, votacao_ciclo.json, cv_ciclo.csv
  - models/historico_decisoes.csv, models/modelo_vigente.json
  - mlflow.db (runs e tags de cada ciclo)
Basta rodar de novo após novos ciclos agendados para o relatório incluí-los:

    pip install reportlab matplotlib pandas pyyaml
    python reports/gerar_relatorio.py
"""
from __future__ import annotations

import json
import sqlite3
from datetime import datetime
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd
import yaml
from reportlab.lib import colors
from reportlab.lib.enums import TA_CENTER, TA_JUSTIFY
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
from reportlab.lib.units import cm
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont
from reportlab.platypus import (Image, KeepTogether, PageBreak, Paragraph, SimpleDocTemplate,
                                Spacer, Table, TableStyle)

RAIZ = Path(__file__).resolve().parents[1]
SAIDA = RAIZ / "reports" / "relatorio_n3.pdf"
TMP = RAIZ / "reports" / "_figuras"
INTEGRANTES = "Miguel, Julia, Ana Olska, Ana Paula e Eduardo"
REPO_URL = "https://github.com/weillerjulia/n3-mlops-bank-marketing"

COR = colors.HexColor("#7B1E3A")
COR_CLARA = colors.HexColor("#F3E6EA")
CINZA = colors.HexColor("#555555")
VERDE = colors.HexColor("#1E6B3A")


# ------------------------------------------------------------------ fontes
def _registrar_fontes() -> tuple[str, str]:
    candidatos = [
        ("C:/Windows/Fonts/arial.ttf", "C:/Windows/Fonts/arialbd.ttf",
         "C:/Windows/Fonts/ariali.ttf"),
        ("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
         "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf",
         "/usr/share/fonts/truetype/dejavu/DejaVuSans-Oblique.ttf"),
    ]
    for normal, negrito, italico in candidatos:
        if Path(normal).exists() and Path(negrito).exists():
            pdfmetrics.registerFont(TTFont("Base", normal))
            pdfmetrics.registerFont(TTFont("Base-Bold", negrito))
            it = "Base"
            if Path(italico).exists():
                pdfmetrics.registerFont(TTFont("Base-Italic", italico))
                it = "Base-Italic"
            pdfmetrics.registerFontFamily("Base", normal="Base", bold="Base-Bold",
                                          italic=it, boldItalic="Base-Bold")
            return "Base", "Base-Bold"
    return "Helvetica", "Helvetica-Bold"


FONTE, FONTE_B = _registrar_fontes()

_ss = getSampleStyleSheet()
S = {
    "corpo": ParagraphStyle("corpo", parent=_ss["Normal"], fontName=FONTE, fontSize=10,
                            leading=14, alignment=TA_JUSTIFY, spaceAfter=6),
    "h1": ParagraphStyle("h1", parent=_ss["Heading1"], fontName=FONTE_B, fontSize=15,
                         textColor=COR, spaceBefore=10, spaceAfter=8),
    "h2": ParagraphStyle("h2", parent=_ss["Heading2"], fontName=FONTE_B, fontSize=11.5,
                         textColor=COR, spaceBefore=8, spaceAfter=4),
    "cel": ParagraphStyle("cel", parent=_ss["Normal"], fontName=FONTE, fontSize=8.5, leading=10.5),
    "celb": ParagraphStyle("celb", parent=_ss["Normal"], fontName=FONTE_B, fontSize=8.5,
                           leading=10.5, textColor=colors.white),
    "leg": ParagraphStyle("leg", parent=_ss["Normal"], fontName=FONTE, fontSize=8.5,
                          textColor=CINZA, alignment=TA_CENTER, spaceAfter=8),
    "capa_t": ParagraphStyle("capa_t", parent=_ss["Title"], fontName=FONTE_B, fontSize=22,
                             textColor=COR, leading=28),
    "capa": ParagraphStyle("capa", parent=_ss["Normal"], fontName=FONTE, fontSize=11.5,
                           alignment=TA_CENTER, leading=17),
    "nota": ParagraphStyle("nota", parent=_ss["Normal"], fontName=FONTE, fontSize=9.5,
                           leading=13, backColor=COR_CLARA, borderPadding=6, spaceBefore=4,
                           spaceAfter=10),
}


def P(txt, est="corpo"):
    return Paragraph(txt, S[est])


def num(v, casas=3):
    if v is None or (isinstance(v, float) and pd.isna(v)):
        return "—"
    return f"{v:,.{casas}f}".replace(",", "X").replace(".", ",").replace("X", ".")


def pct(v, casas=1):
    return num(100 * v, casas) + "%"


def tabela(linhas, larguras, cabecalho=True, destaque=None):
    dados = [[c if not isinstance(c, str) else P(c, "celb" if (cabecalho and i == 0) else "cel")
              for c in linha] for i, linha in enumerate(linhas)]
    t = Table(dados, colWidths=larguras, repeatRows=1 if cabecalho else 0)
    estilo = [
        ("GRID", (0, 0), (-1, -1), 0.4, colors.HexColor("#BBBBBB")),
        ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
        ("TOPPADDING", (0, 0), (-1, -1), 3),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 3),
    ]
    if cabecalho:
        estilo.append(("BACKGROUND", (0, 0), (-1, 0), COR))
        for i in range(1, len(linhas)):
            if i % 2 == 0:
                estilo.append(("BACKGROUND", (0, i), (-1, i), colors.HexColor("#F7F7F7")))
    for i in destaque or []:
        estilo.append(("BACKGROUND", (0, i), (-1, i), colors.HexColor("#E3F1E7")))
    t.setStyle(TableStyle(estilo))
    return t


def figura(caminho, largura_cm, legenda):
    img = Image(str(caminho))
    proporcao = img.imageHeight / img.imageWidth
    img.drawWidth = largura_cm * cm
    img.drawHeight = largura_cm * cm * proporcao
    return KeepTogether([img, P(legenda, "leg")])


# ------------------------------------------------------------------ dados
def carregar():
    cfg = yaml.safe_load((RAIZ / "config/config.yaml").read_text(encoding="utf-8"))
    ciclos = []
    for d in sorted((RAIZ / "evidencias").glob("c*")):
        if not (d / "ciclo_atual.json").exists():
            continue
        c = json.loads((d / "ciclo_atual.json").read_text(encoding="utf-8"))
        c["dir"] = d
        c["votacao"] = json.loads((d / "votacao_ciclo.json").read_text(encoding="utf-8"))
        c["cv"] = pd.read_csv(d / "cv_ciclo.csv")
        ciclos.append(c)
    hist = pd.read_csv(RAIZ / "models/historico_decisoes.csv")
    manifesto = json.loads((RAIZ / "models/modelo_vigente.json").read_text(encoding="utf-8"))
    diag = {}
    for c in ciclos:
        p = RAIZ / "mlartifacts" / c["run_etl"] / "artifacts" / "diagnostico_qualidade.json"
        if p.exists():
            diag[c["cycle_id"]] = json.loads(p.read_text(encoding="utf-8"))
    return cfg, ciclos, hist, manifesto, diag


def runs_mlflow():
    db = RAIZ / "mlflow.db"
    if not db.exists():
        return pd.DataFrame()
    con = sqlite3.connect(db)
    q = """
    SELECT r.run_uuid, r.name, r.status, r.start_time,
           MAX(CASE WHEN t.key='cycle_id' THEN t.value END) AS cycle_id,
           MAX(CASE WHEN t.key='data_version' THEN t.value END) AS data_version,
           MAX(CASE WHEN t.key='etapa' THEN t.value END) AS etapa,
           MAX(CASE WHEN t.key='gatilho' THEN t.value END) AS gatilho
    FROM runs r LEFT JOIN tags t ON t.run_uuid = r.run_uuid
    WHERE r.lifecycle_stage = 'active'
    GROUP BY r.run_uuid ORDER BY r.start_time"""
    df = pd.read_sql(q, con)
    con.close()
    return df


# ------------------------------------------------------------------ figuras
def fig_cv(c, destino):
    cv = c["cv"]
    fig, ax = plt.subplots(figsize=(7, 2.8), dpi=150)
    folds = sorted(cv["fold"].unique())
    larg = 0.38
    for desloc, (mod, cor, rot) in zip(
            (-larg / 2, larg / 2),
            (("A", "#C9A3AF", "A · Regressão Logística"), ("B", "#7B1E3A", "B · Random Forest"))):
        v = cv[cv["modelo"] == mod].sort_values("fold")["average_precision"].values
        ax.bar([f + desloc for f in folds], v, width=larg, color=cor, label=rot)
    prev = cv["prevalencia"].mean()
    ax.axhline(prev, ls="--", color="#555555", lw=1)
    ax.text(folds[-1] + 0.45, prev, " aleatório", va="center", fontsize=7, color="#555555")
    ax.set_xticks(folds, [f"fold {f}" for f in folds])
    ax.set_ylabel("AUC-PR (validação do fold)")
    ax.spines[["top", "right"]].set_visible(False)
    ax.set_ylim(0, cv["average_precision"].max() * 1.3)
    ax.legend(frameon=False, fontsize=8, loc="upper center", ncol=2)
    fig.tight_layout()
    fig.savefig(destino)
    plt.close(fig)


def fig_evolucao(hist, destino):
    h = hist.reset_index(drop=True)
    x = range(1, len(h) + 1)
    fig, ax = plt.subplots(figsize=(7, 2.8), dpi=150)
    ax.plot(x, h["candidato_val_average_precision"], "o-", color="#7B1E3A", label="vencedor do ciclo")
    ax.plot(x, h["vigente_anterior_val_average_precision"], "s--", color="#999999",
            label="vigente anterior (mesma validação)")
    for xi, dec, y in zip(x, h["decisao"], h["candidato_val_average_precision"]):
        ax.annotate("P" if dec == "PROMOVIDO" else "M", (xi, y), textcoords="offset points",
                    xytext=(0, 7), ha="center", fontsize=8,
                    color="#1E6B3A" if dec == "PROMOVIDO" else "#555555")
    ax.set_xticks(list(x), [f"c{i:03d}" for i in x])
    ax.set_ylabel("AUC-PR (validação)")
    ax.spines[["top", "right"]].set_visible(False)
    ax.legend(frameon=False, fontsize=8)
    fig.tight_layout()
    fig.savefig(destino)
    plt.close(fig)


# ------------------------------------------------------------------ cabeçalho/rodapé
def _pagina(canvas, doc):
    if doc.page == 1:
        return
    canvas.saveState()
    canvas.setFont(FONTE, 8)
    canvas.setFillColor(CINZA)
    canvas.drawString(2 * cm, 1.2 * cm, "Católica de Santa Catarina · Análise Preditiva · Trabalho N3")
    canvas.drawRightString(A4[0] - 2 * cm, 1.2 * cm, f"{doc.page}")
    canvas.setStrokeColor(COR)
    canvas.setLineWidth(0.6)
    canvas.line(2 * cm, A4[1] - 1.4 * cm, A4[0] - 2 * cm, A4[1] - 1.4 * cm)
    canvas.restoreState()


# ------------------------------------------------------------------ relatório
def montar():
    cfg, ciclos, hist, manifesto, diag = carregar()
    if not ciclos:
        raise SystemExit("Nenhum ciclo em evidencias/. Rode o pipeline primeiro.")
    TMP.mkdir(exist_ok=True)
    ult = ciclos[-1]
    prim = ciclos[0]
    n = len(ciclos)
    runs = runs_mlflow()
    W = A4[0] - 4 * cm
    st = []

    # ---------------- capa
    st += [Spacer(1, 4 * cm),
           P("Relatório Técnico · Trabalho N3", "capa"),
           Spacer(1, 0.4 * cm),
           P("Pipeline MLOps agendado para previsão de adesão a depósito a prazo (Bank Marketing)",
             "capa_t"),
           Spacer(1, 0.8 * cm),
           P("Votação de features · competição de modelos · promoção controlada · MLflow · "
             "GitHub Actions", "capa"),
           Spacer(1, 2.5 * cm),
           P(f"<b>Integrantes:</b> {INTEGRANTES}", "capa"),
           Spacer(1, 0.3 * cm),
           P("Análise Preditiva · Prof. M. Sc. Rodrigo Hagstrom", "capa"),
           P("Católica de Santa Catarina · Engenharia de Software e Engenharia de IA", "capa"),
           Spacer(1, 0.3 * cm),
           P(f"Repositório: <link href='{REPO_URL}' color='#7B1E3A'>{REPO_URL}</link>", "capa"),
           Spacer(1, 1.5 * cm),
           P(f"Ciclos documentados: {n} (último: {ult['cycle_id']}) · "
             f"gerado em {datetime.now():%d/%m/%Y}", "capa"),
           PageBreak()]

    if n < 2:
        st.append(P(
            "<b>Situação do histórico.</b> Até a geração deste documento, o workflow executou "
            f"{n} ciclo agendado. O enunciado pede histórico de pelo menos dois ciclos: este PDF é "
            "regenerado automaticamente a partir dos artefatos do repositório "
            "(<i>python reports/gerar_relatorio.py</i>) e passa a incluir a comparação entre ciclos "
            "assim que o próximo ciclo agendado for concluído.", "nota"))

    # ---------------- 1. Problema
    lotes = [c["lote"] for c in ciclos]
    taxa = diag.get(prim["cycle_id"], {}).get("taxa_positiva")
    st += [P("1. Problema, dados e avaliação", "h1"),
           P("Um banco português faz campanhas de telemarketing para vender depósitos a prazo. "
             "Ligar para todos os clientes custa caro e a taxa de adesão é baixa. O objetivo é "
             "<b>classificar</b> cada contato pela probabilidade de o cliente contratar o depósito, "
             "para que a equipe ligue primeiro para quem tem mais chance de aderir."),
           tabela([
               ["Item", "Definição"],
               ["Fonte e licença", "Bank Marketing, UCI Machine Learning Repository "
                                   "(arquivo <i>bank-additional-full.csv</i>), licença CC BY 4.0."],
               ["Unidade de observação", "Um contato telefônico com um cliente: 41.188 contatos "
                                         "entre mai/2008 e nov/2010, em ordem cronológica."],
               ["Variável-alvo", "<i>y</i>: contratou o depósito (1) ou não (0). Classe positiva "
                                 "≈ 11% na base completa"
                + (f" e {pct(taxa)} no lote inicial." if taxa else ".")],
               ["Atributos", "Perfil do cliente (idade, profissão, estado civil, escolaridade, "
                             "inadimplência, financiamentos), dados do contato (canal, mês, dia "
                             "da semana, nº de contatos, histórico de campanhas anteriores) e "
                             "indicadores macroeconômicos (taxa de emprego, IPC, confiança do "
                             "consumidor, Euribor 3m, nº de empregados)."],
               ["Métrica principal", "AUC-PR (<i>average_precision</i>), maximizar. Definida em "
                                     "<i>config/config.yaml</i> antes de qualquer treino."],
               ["Particionamento", "70% treino / 15% validação / 15% teste, por hash estável do "
                                   "<i>registro_id</i>; competição A × B por validação cruzada "
                                   f"estratificada de {cfg['avaliacao']['cv_folds']} folds dentro "
                                   "do treino."],
               ["Chegada de dados", "Lote inicial = mai a nov/2008. A cada ciclo, o pipeline "
                                    "busca na fonte os próximos 3 meses de campanha "
                                    "(8 lotes adicionais)."],
           ], [4 * cm, W - 4 * cm], cabecalho=True),
           Spacer(1, 6),
           P("<b>Por que AUC-PR.</b> Com a classe positiva rara, a acurácia premia o modelo que "
             "diz \"não\" para todos, e a AUC-ROC fica otimista porque é dominada pelos "
             "verdadeiros negativos. A AUC-PR mede a qualidade do ranking justamente no topo da "
             "lista, que é onde o banco atua, e não depende de um limiar de decisão. Seu valor de "
             "referência para um modelo aleatório é a própria prevalência, o que dá um piso "
             "objetivo para a regra de promoção. Como métricas secundárias registramos AUC-ROC, "
             "F1, precisão, revocação e o <i>lift</i> nos 20% de clientes com maior "
             "probabilidade (política de contato definida em <i>negocio.percentual_contato</i>)."),
           P("<b>Desenho da avaliação.</b> O hash do <i>registro_id</i> coloca cada contato sempre "
             "na mesma partição, em qualquer ciclo. Assim, quando um lote novo chega, os registros "
             "de validação e teste de ciclos anteriores nunca entram no treino de um modelo "
             "posterior, e o modelo vigente pode ser comparado ao candidato na mesma validação sem "
             "vazamento. O conjunto de teste só é usado para relatar o desempenho final do "
             "vigente; nenhuma decisão (votação, escolha A × B, promoção) olha para ele."),
           P("<b>Ordem temporal.</b> Testamos também um corte temporal para o teste. Como o volume "
             "se concentra em 2008-2009 e o perfil da campanha muda depois, o trecho mais recente "
             "sempre caía em um regime nunca visto, com AUC-ROC ≈ 0,5 em todos os ciclos. Como "
             "cada contato é um evento independente, a ordem do tempo é respeitada na "
             "<b>chegada dos lotes</b> (o modelo nunca vê o futuro antes da hora) e a mudança "
             "de regime é vigiada pelo monitor de drift (seção 10)."),
           ]

    # ---------------- 2. ETL
    d0 = diag.get(prim["cycle_id"], {})
    unk = d0.get("unknown_por_coluna", {})
    st += [P("2. ETL e engenharia de atributos", "h1"),
           P("O notebook <i>01_etl</i> chama <i>src/ingestao.py</i> e "
             "<i>src/preprocessamento.py</i>. A etapa é dividida em duas partes, para evitar "
             "vazamento: transformações <b>determinísticas</b> (que não aprendem nada com os dados) "
             "são aplicadas antes do split; tudo que <b>aprende</b> com os dados fica dentro de um "
             "<i>Pipeline/ColumnTransformer</i> ajustado só no treino."),
           P("Ingestão e versionamento", "h2"),
           P("A fonte completa simula o sistema de origem do banco. A cada ciclo, "
             "<i>ingestao.py</i> identifica os períodos ainda não incorporados, grava o lote em "
             "<i>data/raw/lote_NNN.csv</i> e registra período, nº de linhas e SHA-256 em "
             "<i>data/state/lotes_ingeridos.json</i>. Um lote já registrado nunca é reingerido, e "
             "os registros têm <i>registro_id</i> único, então não há duplicação entre ciclos. "
             "A <i>data_version</i> do ciclo combina o nº do lote com o hash do conjunto "
             f"acumulado (ex.: <i>{prim['data_version']}</i>)."),
           P("Diagnóstico e tratamentos", "h2"),
           tabela([
               ["Verificação", f"Resultado (lote inicial, {num(d0.get('linhas', 0), 0)} linhas)",
                "Tratamento"],
               ["Valores nulos", str(d0.get("nulos", "—")), "Não há nulos explícitos."],
               ["\"unknown\" em categóricas",
                ", ".join(f"{k}: {num(v, 0)}" for k, v in unk.items()) or "—",
                "Mantido como categoria: a falta de informação é informativa "
                "(ex.: <i>default=unknown</i>)."],
               ["Duplicatas exatas", str(d0.get("duplicatas", "—")), "Removidas."],
               ["pdays = 999", "Código mágico para \"nunca contatado\"",
                "Vira NaN + flag <i>contatado_antes</i>; NaN imputado no pipeline."],
               ["Outliers (campaign, previous…)", "Caudas longas",
                "Winsorização nos percentis 1-99 aprendidos só no treino."],
               ["duration", "Só é conhecida após a ligação",
                "Removida (vazamento documentado pela UCI)."],
           ], [3.8 * cm, 6 * cm, W - 9.8 * cm]),
           Spacer(1, 6),
           P("<b>Atributos criados:</b> <i>contatado_antes</i> (flag de campanha anterior), "
             "<i>contatos_totais</i> = campaign + previous (intensidade de contato) e "
             "<i>faixa_etaria</i> (a relação entre idade e adesão não é linear: jovens e "
             "aposentados aderem mais). <b>Pré-processamento aprendido:</b> numéricas → imputação "
             "pela mediana, winsorização e padronização; categóricas → imputação pela moda e one-hot (com <i>handle_unknown='ignore'</i>), todos "
             "dentro do pipeline serializado junto ao modelo."),
           ]

    # ---------------- 3/4. Votação
    tv = pd.DataFrame(ult["votacao"]["tabela_votos"])
    fcfg = cfg["features"]
    st += [P("3. Três métodos de seleção de features", "h1"),
           P("O notebook <i>02_votacao_features</i> executa os três métodos sobre o mesmo treino "
             "já pré-processado (<i>src/features.py</i>, classe <i>VotacaoFeatures</i>). Como o "
             "one-hot gera várias colunas por variável, os resultados são agregados de volta para "
             "a variável original, para que a votação eleja variáveis, e não categorias soltas."),
           tabela([
               ["Método", "Como ranqueia", "Agregação por variável"],
               ["SelectKBest (informação mútua)",
                "Dependência estatística com o alvo; capta relações não lineares e funciona com "
                "colunas numéricas e binárias.", "Maior score entre as colunas"],
               ["Importância por Random Forest",
                "Redução média de impureza (200 árvores, seed 42).", "Soma das importâncias"],
               ["RFE com Regressão Logística",
                "Elimina 10% das colunas por rodada até sobrar uma; devolve o ranking completo.",
                "Melhor posição entre as colunas"],
           ], [4.3 * cm, W - 8.3 * cm, 4 * cm]),
           Spacer(1, 6),
           P(f"Cada método aprova suas <b>top {fcfg['top_k']}</b> variáveis. Os rankings, scores e "
             "o gráfico abaixo são registrados como artefatos do run <i>02_votacao</i> no MLflow."),
           ]
    png = ult["dir"] / "rankings_features.png"
    if png.exists():
        st.append(figura(png, 16, f"Figura 1 – Rankings dos três métodos (ciclo {ult['cycle_id']})."))

    st += [P("4. Votação e eleição das features", "h1"),
           P(f"<b>Regra definida antes da comparação</b> (<i>config.yaml → features</i>): uma "
             f"variável é eleita se aprovada por pelo menos <b>{fcfg['votos_minimos']} dos 3</b> "
             f"métodos. Se a votação eleger menos de <b>{fcfg['min_features']}</b> variáveis, o "
             "conjunto é completado pelas de menor <b>soma de rankings</b>, que também é o "
             "critério de desempate e de ordenação. Na validação cruzada da etapa 5, a votação é "
             "<b>refeita dentro de cada fold</b>, usando só os dados de treino daquele fold — por "
             "isso o nº de features varia entre folds — e o conjunto do ciclo é obtido no treino "
             "completo."),
           ]
    linhas = [["Variável", "SKB rank", "RF rank", "RFE rank", "Votos SKB / RF / RFE", "Total",
               "Σ rank", "Eleita"]]
    destaque = []
    for i, r in enumerate(tv.itertuples(), start=1):
        linhas.append([r.feature, str(r.rank_selectkbest), str(r.rank_random_forest),
                       str(r.rank_rfe),
                       f"{r.voto_selectkbest} / {r.voto_random_forest} / {r.voto_rfe}",
                       str(r.votos), str(r.soma_rankings), "sim" if r.eleita else "não"])
        if r.eleita:
            destaque.append(i)
    st += [tabela(linhas, [3.4 * cm, 1.6 * cm, 1.6 * cm, 1.6 * cm, 3.4 * cm, 1.4 * cm,
                           1.4 * cm, W - 14.4 * cm], destaque=destaque),
           P(f"Tabela de votos do ciclo {ult['cycle_id']} (linhas em verde = eleitas).", "leg"),
           P(f"<b>Conjunto eleito ({len(ult['features_eleitas'])}):</b> "
             + ", ".join(f"<i>{f}</i>" for f in ult["features_eleitas"]) + ". "
             "Esse é o único conjunto usado pelos dois algoritmos no ciclo e é gravado no "
             "manifesto do modelo. Observa-se que os métodos se complementam: a informação mútua "
             "privilegia os indicadores macroeconômicos (Euribor, nº de empregados), a Random "
             "Forest valoriza idade, profissão e escolaridade, e o RFE destaca mês e taxa de "
             "variação do emprego. A regra de 2 de 3 elimina variáveis que só um método apoia, "
             "como <i>nr.employed</i>, muito correlacionada com <i>euribor3m</i>.")]

    if n >= 2:
        st.append(P("Mudanças no conjunto eleito entre ciclos", "h2"))
        lin = [["Ciclo", "Lote", "Nº", "Entraram", "Saíram"]]
        ant = None
        for c in ciclos:
            f = c["features_eleitas"]
            ent = sorted(set(f) - set(ant)) if ant is not None else []
            sai = sorted(set(ant) - set(f)) if ant is not None else []
            lin.append([c["cycle_id"], str(c["lote"]["lote"]), str(len(f)),
                        ", ".join(ent) or ("(inicial)" if ant is None else "—"),
                        ", ".join(sai) or "—"])
            ant = f
        st.append(tabela(lin, [4.2 * cm, 1.2 * cm, 1 * cm, (W - 6.4 * cm) / 2, (W - 6.4 * cm) / 2]))

    # ---------------- 5. Treinamento A x B
    cA, cB = ult["candidatos"]["A"], ult["candidatos"]["B"]
    mA, mB = cfg["modelos"]["A"], cfg["modelos"]["B"]
    st += [P("5. Treinamento dos dois algoritmos", "h1"),
           P("O notebook <i>03_treino_competicao</i> treina os dois candidatos com o "
             "<b>mesmo treino, as mesmas features eleitas, os mesmos folds e a mesma seed</b>. "
             "Cada candidato é um único <i>Pipeline</i> (seleção das features eleitas → "
             "pré-processamento → classificador), o que garante que o pré-processamento seja "
             "reajustado dentro de cada fold e viaje junto com o modelo serializado."),
           tabela([
               ["Modelo", "Algoritmo", "Hiperparâmetros"],
               ["A", "Regressão Logística",
                ", ".join(f"{k}={v}" for k, v in mA["params"].items())],
               ["B", "Random Forest",
                ", ".join(f"{k}={v}" for k, v in mB["params"].items())],
           ], [1.5 * cm, 3.5 * cm, W - 5 * cm]),
           Spacer(1, 8)]
    fig_cv(ult, TMP / "cv.png")
    st.append(figura(TMP / "cv.png", 15.5,
                     f"Figura 2 – AUC-PR por fold, ciclo {ult['cycle_id']}. "
                     "Linha tracejada = prevalência (desempenho de um modelo aleatório)."))
    st += [tabela([
        ["Métrica", "A · Regressão Logística", "B · Random Forest"],
        ["AUC-PR média na CV (± desvio)",
         f"{num(cA['cv_average_precision_media'])} ± {num(cA['cv_average_precision_desvio'])}",
         f"{num(cB['cv_average_precision_media'])} ± {num(cB['cv_average_precision_desvio'])}"],
        ["AUC-PR (validação)", num(cA["val"]["average_precision"]),
         num(cB["val"]["average_precision"])],
        ["AUC-ROC (validação)", num(cA["val"]["roc_auc"]), num(cB["val"]["roc_auc"])],
        ["F1 / Precisão / Revocação",
         f"{num(cA['val']['f1'])} / {num(cA['val']['precision'])} / {num(cA['val']['recall'])}",
         f"{num(cB['val']['f1'])} / {num(cB['val']['precision'])} / {num(cB['val']['recall'])}"],
        ["Lift nos 20% do topo", num(cA["val"]["lift_top"], 2), num(cB["val"]["lift_top"], 2)],
        ["Prevalência na validação", pct(cA["val"]["prevalencia"]), pct(cB["val"]["prevalencia"])],
        ["run_id MLflow", cA["run_id"], cB["run_id"]],
    ], [5 * cm, (W - 5 * cm) / 2, (W - 5 * cm) / 2]),
        P(f"Comparação dos candidatos no ciclo {ult['cycle_id']} "
          f"(data_version {ult['data_version']}).", "leg")]

    # ---------------- 6. Eleição / promoção
    pcfg = cfg["promocao"]
    venc = "B" if cB["cv_average_precision_media"] >= cA["cv_average_precision_media"] else "A"
    st += [P("6. Eleição do vencedor e promoção controlada", "h1"),
           P("O notebook <i>04_selecao_promocao</i> aplica duas decisões automáticas, ambas "
             "definidas em <i>config.yaml</i> antes do primeiro ciclo (<i>src/promocao.py</i>):"),
           P(f"<b>1. Vencedor do ciclo:</b> o candidato com maior AUC-PR média na validação "
             f"cruzada. <b>2. Promoção:</b> o vencedor é retreinado no treino completo e confrontado "
             f"com o modelo vigente na <b>mesma validação</b> do ciclo. Ele só substitui o vigente se "
             f"(a) superar a AUC-PR do vigente em pelo menos <b>{num(pcfg['limiar_melhoria'])}</b> "
             "(margem absoluta, para não trocar de modelo por ruído) e (b) ficar acima da AUC-PR de "
             "um modelo aleatório (a prevalência). Caso contrário, o vigente é mantido. O vigente "
             "anterior é recuperado do estado persistente (manifesto + <i>joblib</i> "
             "versionados no repositório), nunca de arquivos do runner anterior."),
           P(f"No ciclo {ult['cycle_id']}, o vencedor foi o modelo <b>{venc}</b> "
             f"({ult['candidatos'][venc]['algoritmo']}) e a decisão foi "
             f"<b>{ult['decisao']['decisao']}</b>: {ult['decisao']['motivo']}"),
           P("Histórico de decisões", "h2")]
    lin = [["Ciclo", "Gatilho", "Lote (linhas)", "CV A", "CV B", "Vencedor",
            "Val. vencedor", "Val. vigente ant.", "Decisão"]]
    dest = []
    for i, r in enumerate(hist.itertuples(), start=1):
        lin.append([r.cycle_id, r.gatilho,
                    f"{r.lote} ({num(r.linhas_lote, 0)})" if r.lote_novo else "sem lote novo",
                    num(r.cv_A), num(r.cv_B), r.vencedor.split("-", 1)[1].replace("_", " "),
                    num(r.candidato_val_average_precision),
                    num(r.vigente_anterior_val_average_precision), r.decisao])
        if r.decisao == "PROMOVIDO":
            dest.append(i)
    st += [tabela(lin, [3.2 * cm, 1.7 * cm, 2 * cm, 1.1 * cm, 1.1 * cm, 2 * cm, 1.5 * cm,
                        1.6 * cm, W - 14.2 * cm], destaque=dest),
           P("Fonte: <i>models/historico_decisoes.csv</i>. AUC-PR em validação; linhas em verde = "
             "promoção.", "leg")]
    if n >= 2:
        fig_evolucao(hist, TMP / "evolucao.png")
        st.append(figura(TMP / "evolucao.png", 15.5,
                         "Figura 3 – Vencedor de cada ciclo × vigente anterior. "
                         "P = promovido, M = mantido."))

    st += [P("Manifesto do modelo vigente", "h2"),
           P("<i>models/modelo_vigente.json</i> identifica o modelo que está valendo. Os modelos "
             "anteriores são preservados em <i>models/historico/</i> e o manifesto aponta qual "
             "foi substituído."),
           tabela([
               ["Campo", "Valor"],
               ["cycle_id / data_version", f"{manifesto['cycle_id']} / {manifesto['data_version']}"],
               ["Modelo", f"{manifesto['modelo']} · {manifesto['algoritmo']}"],
               ["run_id (MLflow)", manifesto["run_id"]],
               ["Features", ", ".join(manifesto["features"])],
               ["AUC-PR validação / média CV",
                f"{num(manifesto['valor_validacao'])} / {num(manifesto['cv_media'])}"],
               ["Decisão", f"{manifesto['decisao']['decisao']} — {manifesto['decisao']['motivo']}"],
               ["Gatilho / promovido em", f"{manifesto['gatilho']} / {manifesto['promovido_em']}"],
               ["Arquivo", manifesto["arquivo_modelo"]],
               ["Substituiu", manifesto.get("substituiu") or "— (primeiro vigente)"],
           ], [4.5 * cm, W - 4.5 * cm])]

    # ---------------- 7. MLflow
    st += [P("7. MLflow e rastreabilidade", "h1"),
           P(f"Todos os notebooks registram no experimento <b>{cfg['mlflow']['experimento']}</b> "
             "(backend <i>sqlite:///mlflow.db</i>, artefatos em <i>mlartifacts/</i>). Cada ciclo "
             "gera 6 runs que compartilham as tags <i>cycle_id</i>, <i>data_version</i> e "
             "<i>gatilho</i>, e se distinguem pela tag <i>etapa</i>:"),
           tabela([
               ["Run", "Conteúdo registrado"],
               ["01_etl", "Diagnóstico de qualidade (antes/depois), lote, hash, linhas, PSI de drift."],
               ["02_votacao", "Parâmetros da votação, tabela de votos, rankings, gráfico, features eleitas."],
               ["03_modelo_A / 03_modelo_B", "Hiperparâmetros, seed, features, métricas por fold e "
                                             "médias, métricas de validação, modelo serializado."],
               ["04_decisao", "Ranking A × B, métrica do vigente, decisão, motivo, manifesto."],
               ["05_previsoes", "Previsões do vigente (CSV), métricas de impacto da lista de "
                                "contato, modelo/run vigente usado."],
           ], [4.5 * cm, W - 4.5 * cm]),
           Spacer(1, 6),
           P("<b>Persistência entre execuções.</b> O runner do GitHub Actions é descartado ao fim "
             "do job. Por isso, o último passo do workflow commita no próprio repositório o banco "
             "do MLflow, os artefatos, os lotes, o manifesto e os modelos (commit "
             "<i>ciclo cXXX (schedule) [skip ci]</i>), e também publica esse pacote como "
             "<i>artifact</i> baixável (retenção de 90 dias). O ciclo seguinte começa com o "
             "<i>checkout</i> desse estado. Para consultar: "
             "<i>mlflow ui --backend-store-uri sqlite:///mlflow.db</i>; os caminhos dos artefatos "
             "são reescritos para a pasta local por <i>src/utils.py</i>, então o histórico gerado "
             "no GitHub abre normalmente em qualquer máquina.")]
    if not runs.empty:
        r = runs.dropna(subset=["cycle_id"])
        lin = [["cycle_id", "etapa", "run_id", "status", "gatilho"]]
        for x in r.itertuples():
            lin.append([x.cycle_id, x.etapa or x.name, x.run_uuid, x.status, x.gatilho or "—"])
        st += [tabela(lin, [3.3 * cm, 3 * cm, 6.3 * cm, 2.1 * cm, W - 14.7 * cm]),
               P("Runs lidos diretamente de <i>mlflow.db</i>.", "leg")]

    # ---------------- 8. Workflow
    st += [P("8. Arquitetura do fluxo único e execução agendada", "h1"),
           P("Há um único workflow, <i>.github/workflows/pipeline_mlops.yml</i>, com um único job "
             "sequencial. O <i>cron</i> <b>\"17 11 * * *\"</b> dispara todo dia às 08:17 (Brasília) — "
             "minuto \"quebrado\" para fugir do pico de agendamentos da hora cheia. "
             "<i>workflow_dispatch</i> existe só para depuração e <i>concurrency</i> impede dois "
             "ciclos simultâneos, já que o estado é sequencial.")]
    etapas = [
        ("1", "schedule (cron)", "GitHub Actions"),
        ("2", "Checkout do estado persistente + instalação", "workflow"),
        ("3", "Ingestão do próximo lote, data_version, ETL, drift", "01_etl"),
        ("4-5", "3 métodos de seleção + votação", "02_votacao_features"),
        ("6", "Treino e CV de A (Reg. Logística) e B (Random Forest)", "03_treino_competicao"),
        ("7-9", "Vencedor do ciclo × vigente → promove ou mantém", "04_selecao_promocao"),
        ("10", "Previsões com o vigente + comparação de ciclos", "05_previsoes"),
        ("—", "Commit do estado + upload do artifact do ciclo", "workflow"),
    ]
    lin = [["Passo", "O que acontece", "Onde"]] + [[a, b, c] for a, b, c in etapas]
    st += [tabela(lin, [1.6 * cm, W - 6.1 * cm, 4.5 * cm]),
           P("Todos os passos de 3 a 10 compartilham o mesmo <i>cycle_id</i> e "
             "<i>data_version</i>, gravados em <i>data/state/ciclo_atual.json</i> pelo notebook 01.",
             "leg"),
           P("<b>Falhas interrompem o fluxo.</b> <i>executar_pipeline.sh</i> roda com "
             "<i>set -euo pipefail</i> e chama <i>jupyter nbconvert --execute</i> notebook a "
             "notebook; qualquer exceção encerra o job antes do passo que salva o estado. Como a "
             "promoção só passa a valer quando o manifesto e o modelo são commitados, uma falha "
             "nunca deixa um modelo promovido indevidamente: o vigente anterior continua no "
             "repositório."),
           P("Evidências de execução", "h2")]
    lin = [["Ciclo", "Gatilho", "Início (UTC)", "Execução no GitHub Actions"]]
    for c in ciclos:
        e = c["execucao"]
        url = e.get("github_run_url")
        lin.append([c["cycle_id"], e["gatilho"], e["iniciado_em"].replace("T", " ")[:19],
                    f"<link href='{url}' color='#7B1E3A'>run {e.get('github_run_id')}</link>"
                    if url else "local"])
    st += [tabela(lin, [3.8 * cm, 2.4 * cm, 3.8 * cm, W - 10 * cm]),
           Spacer(1, 6),
           P("O campo <i>gatilho</i> vem de <i>GITHUB_EVENT_NAME</i> e é gravado no contexto do "
             "ciclo, nas tags dos runs do MLflow, no manifesto e no histórico de decisões, o que "
             "comprova que a execução foi disparada pelo <b>schedule</b>, e não manualmente. "
             "Os notebooks executados de cada ciclo, com todas as saídas e logs, ficam em "
             "<i>evidencias/&lt;cycle_id&gt;/</i>.")]

    # ---------------- 9. Previsões
    prev = pd.read_csv(ult["dir"] / "previsoes.csv")
    h_ult = hist.iloc[-1]
    st += [P("9. Previsões com o modelo vigente", "h1"),
           P("O notebook <i>05_previsoes</i> carrega <i>models/vigente/modelo.joblib</i> — um único "
             "<i>Pipeline</i> com seleção das features eleitas, pré-processamento ajustado e "
             "classificador — e confere que as features do manifesto são as mesmas do pipeline "
             "carregado (<i>assert</i> que interrompe o ciclo se divergirem). O desempenho final "
             "do vigente é medido no <b>conjunto de teste</b>, reservado e nunca usado nas "
             "decisões, e gravado no histórico. Resultado do ciclo:"),
           P(f"• desempenho de teste do vigente: AUC-PR = <b>{num(h_ult['vigente_teste_average_precision'])}</b> e lift de "
             f"<b>{num(h_ult['vigente_teste_lift_top'], 2)}×</b> nos 20% de maior probabilidade, ou "
             "seja, ligar para esse grupo gera esse múltiplo de adesões em relação a ligar ao "
             "acaso;"),
           P(f"• o notebook gera previsões para os <b>{num(len(prev), 0)}</b> contatos do <b>próximo período "
             "de campanha</b> (registros que o pipeline ainda não incorporou, logo não usados na "
             "escolha do modelo), ordenados por probabilidade e com a recomendação "
             f"<i>ligar</i> para os {pct(cfg['negocio']['percentual_contato'], 0)} do topo;"),
           P("• e mostra o impacto da lista (adesões capturadas ligando só para o topo × ligando "
             "para todos) e a comparação entre ciclos (lote, features, CV de A e B, decisão), "
             "além de todos os runs do experimento recuperados do MLflow.")]
    top = prev.head(8)
    lin = [["registro_id", "período", "idade", "profissão", "canal", "prob. adesão", "ranking", "ligar"]]
    for r in top.itertuples():
        lin.append([str(r.registro_id), r.periodo, str(r.age), r.job, r.contact,
                    num(r.probabilidade_adesao, 3), str(r.ranking), "sim" if r.ligar else "não"])
    st += [tabela(lin, [2.2 * cm, 1.8 * cm, 1.4 * cm, 2.6 * cm, 2.2 * cm, 2.4 * cm, 1.7 * cm,
                        W - 14.3 * cm]),
           P(f"Primeiras linhas de <i>evidencias/{ult['cycle_id']}/previsoes.csv</i>.", "leg")]

    # ---------------- 10. Critério 10
    psis = hist["psi_max"].dropna()
    alertas = hist[hist["drift_alerta"].astype(str) == "True"]
    st += [P("10. Recurso adicional: monitoramento de drift com alerta", "h1"),
           tabela([
               ["Problema identificado",
                "Os dados cobrem 2008-2010: crise financeira, queda da Euribor de ~5% para ~1% e "
                "mudança no público da campanha. A prevalência de adesão muda de ~5% no lote "
                "inicial para valores bem maiores depois. Um modelo treinado no regime antigo pode "
                "envelhecer sem que ninguém perceba, e a regra de promoção sozinha só olha "
                "desempenho, não a causa."],
               ["Justificativa",
                "Saber <i>quais</i> atributos mudaram explica por que a votação trocou features ou "
                "por que o vigente perdeu desempenho, e avisa a equipe antes de o problema chegar "
                "às ligações."],
               ["Solução implementada",
                "<i>src/drift.py</i> calcula o PSI (Population Stability Index) de cada atributo, "
                "lote novo × dados já conhecidos, com 10 bins por quantis para numéricos e "
                "frequências para categóricos. Faixas: PSI &lt; 0,10 estável; 0,10–0,25 atenção; "
                "≥ 0,25 <b>alerta</b> (limiares em <i>config.yaml → drift</i>)."],
               ["Ponto de integração",
                "Executado no notebook 01 a cada lote novo. O alerta vira anotação "
                "<i>::warning::</i> no GitHub Actions e tabela no resumo da execução; o PSI por "
                "atributo é registrado no run <i>01_etl</i> do MLflow; a flag "
                "<i>drift_alerta</i> e as features afetadas vão para o manifesto do modelo e para "
                "<i>historico_decisoes.csv</i>."],
               ["Resultado demonstrado",
                (f"{len(alertas)} ciclo(s) com alerta; PSI máximo observado = {num(psis.max())}. "
                 if len(psis) else
                 "No ciclo inicial não há referência anterior, então o PSI ainda não é calculado "
                 "(<i>psi_max</i> vazio no histórico). ")
                + "A partir do segundo ciclo, a tabela de PSI por atributo e o gráfico ficam no "
                  "notebook 01 executado em <i>evidencias/&lt;cycle_id&gt;/</i>, e a coluna "
                  "<i>drift_alerta</i> do histórico mostra em quais ciclos houve alerta."],
           ], [3.8 * cm, W - 3.8 * cm], cabecalho=False)]
    if len(psis):
        lin = [["Ciclo", "PSI máx.", "Alerta"]]
        for r in hist.itertuples():
            lin.append([r.cycle_id, num(r.psi_max), "sim" if str(r.drift_alerta) == "True" else "não"])
        st += [Spacer(1, 6), tabela(lin, [5 * cm, 3 * cm, 3 * cm])]

    # ---------------- 11. Ensaio local
    st += [P("11. Ensaio local de referência (9 ciclos)", "h1"),
           P("Antes de publicar o repositório, o pipeline completo foi executado localmente "
             "(<i>bash executar_pipeline.sh</i>, gatilho <i>local</i>) até esgotar a fonte, para "
             "validar o comportamento de longo prazo. Esses resultados não substituem as "
             "evidências agendadas, mas mostram o que se espera dos próximos ciclos:"),
           tabela([
               ["Ciclo", "Lote", "Registros novos", "AUC-PR candidato", "AUC-PR vigente ant.", "Decisão"],
               ["1", "mai–nov/2008", "27.680", "0,104", "—", "PROMOVIDO (RF)"],
               ["2", "dez/08–abr/09", "2.750", "0,204", "0,107", "PROMOVIDO"],
               ["3", "mai–jul/09", "6.687", "0,256", "0,222", "PROMOVIDO"],
               ["4", "ago–out/09", "1.484", "0,321", "0,282", "PROMOVIDO"],
               ["5", "nov/09–mar/10", "793", "0,363", "0,347", "PROMOVIDO"],
               ["6", "abr–jun/10", "615", "0,415", "0,390", "PROMOVIDO"],
               ["7", "jul–set/10", "847", "0,458", "0,451", "PROMOVIDO"],
               ["8", "out–nov/10", "332", "0,463", "0,467", "MANTIDO"],
               ["9", "(sem lote novo)", "0", "0,463", "0,467", "MANTIDO"],
           ], [1.4 * cm, 3 * cm, 2.7 * cm, 3 * cm, 3.2 * cm, W - 13.3 * cm]),
           Spacer(1, 6),
           P("Os ciclos 8 e 9 mostram a regra de promoção funcionando no sentido contrário: o "
             "candidato ficou abaixo do vigente e o vigente foi preservado automaticamente.")]

    # ---------------- 12. Limitações e conclusões
    st += [P("12. Limitações", "h1"),
           P("• <b>Chegada simulada:</b> os lotes vêm de uma base histórica fechada; em produção a "
             "ingestão leria o CRM do banco."),
           P("• <b>Sinal fraco no início:</b> o lote de 2008 tem "
             "prevalência baixa e pouca variação macroeconômica, então o primeiro modelo fica "
             f"pouco acima do aleatório (AUC-PR de teste {num(hist.iloc[0]['vigente_teste_average_precision'])}). "
             "O desempenho cresce com os lotes de 2009-2010."),
           P("• <b>Particionamento por hash:</b> "
             "mede generalização para contatos não vistos do mesmo período, não para o futuro; o "
             "drift mitiga, mas não substitui, uma avaliação temporal."),
           P("• <b>Estado no Git:</b> "
             "versionar <i>mlflow.db</i> e modelos no repositório é simples e gratuito, mas cresce "
             "a cada ciclo; com mais volume, o caminho seria um servidor MLflow com armazenamento "
             "de objetos."),
           P("• <b>Agendamento do GitHub:</b> pode atrasar execuções e desativa "
             "schedules após 60 dias sem atividade."),
           P("• <b>Hiperparâmetros fixos:</b> não há busca "
             "de hiperparâmetros por ciclo, para manter a comparação entre ciclos estável e o "
             "tempo de execução baixo."),
           P("13. Conclusões", "h1"),
           P("O projeto entrega um pipeline MLOps único, agendado e reprodutível: a cada execução "
             "do <i>cron</i>, um lote novo é ingerido sem duplicação, versionado e diagnosticado; "
             "três métodos votam um conjunto único de features; Regressão Logística e Random "
             "Forest competem com o mesmo conjunto e o mesmo protocolo; o vencedor só substitui o "
             "vigente se superar uma margem definida antes do primeiro treino; e tudo fica "
             "registrado no MLflow com <i>cycle_id</i> e <i>data_version</i> comuns, recuperável "
             "entre execuções. O conjunto de teste permanece fora de todas as decisões, e o "
             "monitor de drift explica as mudanças de regime que o modelo enfrenta ao longo das "
             "campanhas."),
           ]

    doc = SimpleDocTemplate(str(SAIDA), pagesize=A4, leftMargin=2 * cm, rightMargin=2 * cm,
                            topMargin=2 * cm, bottomMargin=2 * cm,
                            title="Relatório Técnico N3 - Pipeline MLOps Bank Marketing",
                            author=INTEGRANTES)
    doc.build(st, onFirstPage=_pagina, onLaterPages=_pagina)
    for f in TMP.glob("*.png"):
        f.unlink()
    TMP.rmdir()
    print(f"Gerado: {SAIDA.relative_to(RAIZ)} ({n} ciclo(s))")


if __name__ == "__main__":
    montar()
