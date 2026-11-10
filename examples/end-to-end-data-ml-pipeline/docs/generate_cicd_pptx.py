#!/usr/bin/env python3
"""PPTX PT-BR — artefatos gerados (além dos models) nos pipelines DataOps/MLOps/Promote."""

from pathlib import Path

from pptx import Presentation
from pptx.dml.color import RGBColor
from pptx.enum.shapes import MSO_SHAPE
from pptx.enum.text import PP_ALIGN
from pptx.util import Inches, Pt

DOCS = Path(__file__).resolve().parent
OUT_DOCS = DOCS / "CICD-DataOps-MLOps-Promote-PT-v2.pptx"
OUT_STACK = Path("/mnt/c/Users/guslc/OneDrive/Documentos/stack/CICD-DataOps-MLOps-Promote-PT-v2.pptx")

NAVY = RGBColor(0x0B, 0x2D, 0x4A)
TEAL = RGBColor(0x0D, 0x7A, 0x8C)
ACCENT = RGBColor(0xE8, 0x7A, 0x2E)
GREEN = RGBColor(0x1B, 0x7A, 0x4E)
WHITE = RGBColor(0xFF, 0xFF, 0xFF)
DARK = RGBColor(0x1A, 0x1A, 0x1A)
MUTED = RGBColor(0x4A, 0x55, 0x68)
LIGHT = RGBColor(0xF4, 0xF7, 0xFA)
SOFT = RGBColor(0xDCE, 0xE5, 0xEE) if False else RGBColor(0xDC, 0xE5, 0xEE)


def _run(p, text, size=15, bold=False, color=DARK):
    r = p.add_run()
    r.text = text
    r.font.size = Pt(size)
    r.font.bold = bold
    r.font.color.rgb = color
    r.font.name = "Calibri"
    return r


def _fill(shape, color):
    shape.fill.solid()
    shape.fill.fore_color.rgb = color
    shape.line.fill.background()


def _bg(slide, color=LIGHT):
    s = slide.shapes.add_shape(MSO_SHAPE.RECTANGLE, 0, 0, Inches(13.333), Inches(7.5))
    _fill(s, color)
    tree = slide.shapes._spTree
    el = s._element
    tree.remove(el)
    tree.insert(2, el)


def _header(slide, title, subtitle=None):
    bar = slide.shapes.add_shape(MSO_SHAPE.RECTANGLE, 0, 0, Inches(13.333), Inches(1.05))
    _fill(bar, NAVY)
    t = slide.shapes.add_textbox(Inches(0.5), Inches(0.18), Inches(12.3), Inches(0.5))
    _run(t.text_frame.paragraphs[0], title, size=22, bold=True, color=WHITE)
    if subtitle:
        s = slide.shapes.add_textbox(Inches(0.5), Inches(0.65), Inches(12.3), Inches(0.35))
        _run(s.text_frame.paragraphs[0], subtitle, size=11, color=SOFT)


def _bullets(slide, items, left=0.55, top=1.25, width=12.2, height=5.9, size=14):
    box = slide.shapes.add_textbox(Inches(left), Inches(top), Inches(width), Inches(height))
    tf = box.text_frame
    tf.word_wrap = True
    for i, item in enumerate(items):
        if isinstance(item, str):
            text, bold, color, sz, plain = item, False, DARK, size, False
        else:
            text = item["text"]
            bold = item.get("bold", False)
            color = item.get("color", DARK)
            sz = item.get("size", size)
            plain = item.get("plain", False)
        p = tf.paragraphs[0] if i == 0 else tf.add_paragraph()
        p.space_after = Pt(5)
        prefix = "" if plain else "• "
        _run(p, prefix + text, size=sz, bold=bold, color=color)


def title_slide(prs, title, subtitle, foot):
    slide = prs.slides.add_slide(prs.slide_layouts[6])
    _bg(slide, NAVY)
    accent = slide.shapes.add_shape(
        MSO_SHAPE.RECTANGLE, 0, Inches(5.9), Inches(13.333), Inches(1.6)
    )
    _fill(accent, TEAL)
    box = slide.shapes.add_textbox(Inches(0.7), Inches(2.0), Inches(12), Inches(2.5))
    tf = box.text_frame
    tf.word_wrap = True
    _run(tf.paragraphs[0], title, size=32, bold=True, color=WHITE)
    sub = slide.shapes.add_textbox(Inches(0.7), Inches(4.5), Inches(12), Inches(1.0))
    tf = sub.text_frame
    tf.word_wrap = True
    _run(tf.paragraphs[0], subtitle, size=16, color=WHITE)
    f = slide.shapes.add_textbox(Inches(0.7), Inches(6.35), Inches(12), Inches(0.7))
    _run(f.text_frame.paragraphs[0], foot, size=12, color=WHITE)


def section(prs, num, title, blurb):
    slide = prs.slides.add_slide(prs.slide_layouts[6])
    _bg(slide, LIGHT)
    left = slide.shapes.add_shape(MSO_SHAPE.RECTANGLE, 0, 0, Inches(3.8), Inches(7.5))
    _fill(left, TEAL)
    n = slide.shapes.add_textbox(Inches(0.2), Inches(2.8), Inches(3.4), Inches(1.2))
    p = n.text_frame.paragraphs[0]
    p.alignment = PP_ALIGN.CENTER
    _run(p, num, size=56, bold=True, color=WHITE)
    t = slide.shapes.add_textbox(Inches(4.3), Inches(2.5), Inches(8.4), Inches(1.3))
    _run(t.text_frame.paragraphs[0], title, size=28, bold=True, color=NAVY)
    b = slide.shapes.add_textbox(Inches(4.3), Inches(4.0), Inches(8.4), Inches(2.3))
    tf = b.text_frame
    tf.word_wrap = True
    _run(tf.paragraphs[0], blurb, size=15, color=MUTED)


def content(prs, title, items, subtitle=None):
    slide = prs.slides.add_slide(prs.slide_layouts[6])
    _bg(slide, LIGHT)
    _header(slide, title, subtitle)
    _bullets(slide, items)


def cards3(prs, title, cards, subtitle=None):
    """cards: [(heading, [lines...]), ...] up to 3"""
    slide = prs.slides.add_slide(prs.slide_layouts[6])
    _bg(slide, LIGHT)
    _header(slide, title, subtitle)
    width = 3.95
    gap = 0.22
    start = 0.5
    colors = [TEAL, NAVY, ACCENT]
    for i, (heading, lines) in enumerate(cards[:3]):
        x = start + i * (width + gap)
        card = slide.shapes.add_shape(
            MSO_SHAPE.ROUNDED_RECTANGLE, Inches(x), Inches(1.3), Inches(width), Inches(5.8)
        )
        _fill(card, WHITE)
        stripe = slide.shapes.add_shape(
            MSO_SHAPE.RECTANGLE, Inches(x), Inches(1.3), Inches(width), Inches(0.12)
        )
        _fill(stripe, colors[i % 3])
        h = slide.shapes.add_textbox(Inches(x + 0.18), Inches(1.55), Inches(width - 0.35), Inches(0.55))
        tf = h.text_frame
        tf.word_wrap = True
        _run(tf.paragraphs[0], heading, size=13, bold=True, color=NAVY)
        body = slide.shapes.add_textbox(
            Inches(x + 0.18), Inches(2.2), Inches(width - 0.35), Inches(4.7)
        )
        tf = body.text_frame
        tf.word_wrap = True
        for j, line in enumerate(lines):
            p = tf.paragraphs[0] if j == 0 else tf.add_paragraph()
            p.space_after = Pt(4)
            _run(p, line, size=11, color=DARK)


def two_col(prs, title, ltitle, litems, rtitle, ritems, subtitle=None):
    slide = prs.slides.add_slide(prs.slide_layouts[6])
    _bg(slide, LIGHT)
    _header(slide, title, subtitle)
    for x, ttl, items, color in (
        (0.45, ltitle, litems, TEAL),
        (6.85, rtitle, ritems, ACCENT),
    ):
        card = slide.shapes.add_shape(
            MSO_SHAPE.ROUNDED_RECTANGLE, Inches(x), Inches(1.3), Inches(5.95), Inches(5.8)
        )
        _fill(card, WHITE)
        ht = slide.shapes.add_textbox(Inches(x + 0.22), Inches(1.45), Inches(5.5), Inches(0.4))
        _run(ht.text_frame.paragraphs[0], ttl, size=15, bold=True, color=color)
        body = slide.shapes.add_textbox(Inches(x + 0.22), Inches(2.0), Inches(5.5), Inches(4.9))
        tf = body.text_frame
        tf.word_wrap = True
        for i, text in enumerate(items):
            p = tf.paragraphs[0] if i == 0 else tf.add_paragraph()
            p.space_after = Pt(6)
            _run(p, "• " + text, size=12, color=DARK)


def build():
    prs = Presentation()
    prs.slide_width = Inches(13.333)
    prs.slide_height = Inches(7.5)

    # 1
    title_slide(
        prs,
        "Artefatos gerados pelos pipelines\n(além dos Models)",
        "Mapa completo do que DataOps, MLOps e Promote gravam no S3, Athena, MLflow e infra",
        "CI/CD SageMaker Unified Studio  ·  v2  ·  foco em dados, qualidade, features, reports e serving",
    )

    # 2
    content(
        prs,
        "Por que falar de artefatos (não só Models)?",
        [
            {"text": "Models na UI são só a ponta — a linhagem mora em S3 + Catálogo + reports", "bold": True},
            "DataOps publica tabelas e evidência de qualidade que o treino consome",
            "MLOps publica features, baselines de drift, métricas e predições enriquecidas",
            "Promote publica endpoints e artefatos de serving — mas o contrato de dados já veio antes",
            "Operação, auditoria e rollback dependem desses arquivos, não do nome do Model na tela",
            {"text": "Este deck: inventário por pasta / tabela / JSON — o que é, quem gera, quem consome", "bold": True, "color": TEAL},
        ],
        subtitle="Escopo: e2e-dataops-pipeline · e2e-mlops-pipeline · e2e-mlops-promote",
    )

    # 3 — mapa mestre
    cards3(
        prs,
        "Mapa mestre dos artefatos (sem Models)",
        [
            (
                "DataOps → dados & DQ",
                [
                    "CSV bruto (input)",
                    "staging/ (Parquet)",
                    "transformed/ (Parquet)",
                    "campaign_results (Athena)",
                    "quality_reports/ + dq_reports",
                    "athena_results/",
                    "código Glue + DAG YAML",
                ],
            ),
            (
                "MLOps → features & evidência",
                [
                    "features/train|val|test (CSV)",
                    "features/*_parquet + tabelas Glue",
                    "drift_reference/",
                    "drift_reports/*/result.json",
                    "evaluation_report.json",
                    "predictions_enriched (Delta)",
                    "runs MLflow (métricas/tags)",
                ],
            ),
            (
                "Promote → runtime & trilha",
                [
                    "deployments/ reports JSON",
                    "realtime-serving/…/tar.gz",
                    "  (code/inference.py + meta)",
                    "endpoint + configs (runtime)",
                    "CustomerMetadata no package",
                    "Issues GHA (gates)",
                    "EventBridge + Lambda trigger",
                ],
            ),
        ],
        subtitle="Tudo sob Shared do projeto SMUS, salvo Registry/infra AWS gerenciada",
    )

    # 4 — onde ver
    two_col(
        prs,
        "Onde encontrar no Studio",
        "Files → Shared → stack/",
        [
            "Bucket do projeto (project.s3_default_folder)",
            "data_pipeline/ — ETL + outputs DataOps",
            "mlops_pipeline/ — features, eval, drift, deploy",
            "workflows/ — config + dags Airflow Serverless",
            "Paths resolvidos em deploy: {proj…s3_shared.s3Uri}",
        ],
        "Query Editor / outros",
        [
            "AwsDataCatalog → stack_bank_mktg_dev",
            "Tabelas: campaign_results, dq_reports",
            "Também: features_train|validation|test",
            "MLflow → experimento stack-experiments",
            "Inference endpoints → stack-prediction-<stage>",
        ],
        subtitle="Projeto exemplo: e2e-data-stack-ml-ops-dev",
    )

    # ========== DATAOPS ARTIFACTS ==========
    section(
        prs,
        "01",
        "Artefatos DataOps",
        "Gerados por e2e-dataops-pipeline → DAG data_pipeline (hoje só no projeto de desenvolvimento).",
    )

    cards3(
        prs,
        "DataOps — cadeia que produz cada artefato",
        [
            (
                "1. Ingest",
                [
                    "Job: stack-ingest (Glue)",
                    "Lê: data/stack-sample.csv",
                    "Escreve: output/staging/",
                    "Formato: Parquet",
                    "Papel: tipagem + limpeza",
                    "  inicial do CSV (;)",
                ],
            ),
            (
                "2. Transform",
                [
                    "Job: stack-transform",
                    "Lê: staging/",
                    "Escreve: output/transformed/",
                    "Formato: Parquet",
                    "Deriva: age_group,",
                    "  high_contact, prev_success",
                    "Filtra job=unknown",
                ],
            ),
            (
                "3. Quality + Athena",
                [
                    "Job: stack-data-quality",
                    "Escreve: quality_reports/<ds>/",
                    "Athena: CREATE DATABASE",
                    "CREATE TABLE campaign_results",
                    "MSCK REPAIR TABLE",
                    "Logs SQL → athena_results/",
                ],
            ),
        ],
        subtitle="Ordem no DAG: ingest → transform → DQ → create DB/table → refresh",
    )

    content(
        prs,
        "Artefato: CSV de entrada (stack-sample.csv)",
        [
            {"text": "Caminho", "bold": True, "color": TEAL, "plain": True, "size": 16},
            "shared/stack/data_pipeline/data/stack-sample.csv",
            {"text": "O que é", "bold": True, "color": TEAL, "plain": True, "size": 16},
            "Dataset bank marketing (~41k linhas), separado por “;”, com header",
            "Publicado no deploy do DataOps (storage item data/ no manifesto)",
            {"text": "Consumidores", "bold": True, "color": TEAL, "plain": True, "size": 16},
            "Glue ingest (--source_path) e, depois, drift check (CURRENT_PATH) + snapshot drift_reference",
            {"text": "Implicação CI", "bold": True, "color": ACCENT, "plain": True, "size": 16},
            "Push que altera o CSV dispara e2e-dataops → refresh de staging/transformed/tabelas",
        ],
    )

    content(
        prs,
        "Artefato: output/staging/ (Parquet intermediário)",
        [
            {"text": "Caminho", "bold": True, "color": TEAL, "plain": True, "size": 16},
            "…/stack/data_pipeline/output/staging/",
            {"text": "Gerador", "bold": True, "color": TEAL, "plain": True, "size": 16},
            "glue_ingest_raw_data.py — lê CSV, sanitiza colunas, grava Parquet (overwrite, coalesce 1)",
            {"text": "Papel", "bold": True, "color": TEAL, "plain": True, "size": 16},
            "Camada bronze tipada: isola parsing do CSV do restante do ETL",
            "Não é a tabela Athena — é input exclusivo do transform",
            {"text": "Operação", "bold": True, "color": MUTED, "plain": True, "size": 16},
            "Sobrescrito a cada run; útil para debug se o transform falhar",
        ],
    )

    content(
        prs,
        "Artefato: output/transformed/ → LOCATION de campaign_results",
        [
            {"text": "Caminho S3", "bold": True, "color": TEAL, "plain": True, "size": 16},
            "…/stack/data_pipeline/output/transformed/  (Parquet)",
            {"text": "Gerador", "bold": True, "color": TEAL, "plain": True, "size": 16},
            "glue_transform_aggregate.py — enriquece staging e escreve analytics_path",
            {"text": "Colunas derivadas (além do schema original)", "bold": True, "color": TEAL, "plain": True, "size": 16},
            "age_group (young/middle/senior) · high_contact (campaign>3) · prev_success (poutcome)",
            "Remove linhas com job=unknown",
            {"text": "Elo Athena", "bold": True, "color": TEAL, "plain": True, "size": 16},
            "CREATE EXTERNAL TABLE … campaign_results … LOCATION '<esse prefixo>' STORED AS PARQUET",
            {"text": "Consumidor principal", "bold": True, "color": ACCENT, "plain": True, "size": 16},
            "MLOps feature_engineering (INPUT_PATH / tabela de catálogo) — base do treino",
        ],
    )

    content(
        prs,
        "Artefato Athena: campaign_results",
        [
            {"text": "Identidade", "bold": True, "color": TEAL, "plain": True, "size": 16},
            "AwsDataCatalog · database stack_bank_mktg_dev (ATHENA_DB_NAME) · table campaign_results",
            {"text": "Como ver", "bold": True, "color": TEAL, "plain": True, "size": 16},
            "Query Editor: SELECT * FROM stack_bank_mktg_dev.campaign_results LIMIT 10",
            {"text": "Schema declarado (CREATE TABLE)", "bold": True, "color": TEAL, "plain": True, "size": 16},
            "Atributos demográficos/campanha + macros (emp_var_rate, euribor3m…) + alvo y",
            "Campos engenheirados expostos na tabela: age_group, contacted_before (manifesto SQL)",
            {"text": "Ciclo de vida", "bold": True, "color": TEAL, "plain": True, "size": 16},
            "CREATE DATABASE IF NOT EXISTS → CREATE EXTERNAL TABLE IF NOT EXISTS → MSCK REPAIR",
            "Dados físicos = arquivos em transformed/; a tabela é só metadado + SQL",
            {"text": "Escopo do workflow", "bold": True, "color": ACCENT, "plain": True, "size": 16},
            "DataOps CI hoje só atualiza isso em DEV — não promove dataset para test/prod",
        ],
    )

    content(
        prs,
        "Artefato: quality_reports/ + tabela dq_reports",
        [
            {"text": "S3", "bold": True, "color": TEAL, "plain": True, "size": 16},
            "…/output/quality_reports/<report_date>/  → Parquet com uma linha por check",
            {"text": "Catálogo (Studio)", "bold": True, "color": TEAL, "plain": True, "size": 16},
            "stack_bank_mktg_dev.dq_reports — consultável no Query Editor",
            {"text": "Schema", "bold": True, "color": TEAL, "plain": True, "size": 16},
            "check_type | column_name | metric_value | threshold | status",
            {"text": "Checks", "bold": True, "color": TEAL, "plain": True, "size": 16},
            "null_rate (<10%) · duplicates (=0) · row_count (>0) · range age 17–100 · y ∈ {yes,no}",
            {"text": "Semântica de status", "bold": True, "color": TEAL, "plain": True, "size": 16},
            "PASS / WARN / FAIL — FAIL aborta o Glue job e derruba o DAG data_pipeline",
            {"text": "Exemplo", "bold": True, "color": MUTED, "plain": True, "size": 16},
            "SELECT * FROM stack_bank_mktg_dev.dq_reports ORDER BY status, check_type;",
        ],
    )

    content(
        prs,
        "Artefato auxiliar: athena_results/ + código/DAG",
        [
            {"text": "athena_results/", "bold": True, "color": TEAL, "plain": True, "size": 16},
            "output_location das queries Athena (CREATE/MSCK) — staging de resultado SQL, não dado de negócio",
            {"text": "Código versionado no Shared", "bold": True, "color": TEAL, "plain": True, "size": 16},
            "stack/data_pipeline/src/glue-jobs/*.py — scripts que o GlueJobOperator aponta via script_location",
            "stack/data_pipeline/workflows/data_pipeline.yaml — definição do DAG",
            "Shared/workflows/dags — cópia operacional do Airflow Serverless após CLI deploy",
            {"text": "Jobs Glue nomeados", "bold": True, "color": TEAL, "plain": True, "size": 16},
            "stack-ingest · stack-transform · stack-data-quality (update_config=true a cada run)",
        ],
    )

    # ========== MLOPS ARTIFACTS ==========
    section(
        prs,
        "02",
        "Artefatos MLOps (treino / drift / eval)",
        "Gerados principalmente em DEV por training_pipeline e drift_check_pipeline — não são Models da UI.",
    )

    cards3(
        prs,
        "MLOps — pastas sob stack/mlops_pipeline/",
        [
            (
                "src/ + workflows/",
                [
                    "Scripts Processing/Glue",
                    "inference.py + feature_schema",
                    "train_model_sourcedir.tar.gz",
                    "DAGs: training / deploy /",
                    "  drift_check",
                    "Notebooks validate_*.ipynb",
                ],
            ),
            (
                "data/bank_mktg/",
                [
                    "features/… (splits)",
                    "drift_reference/",
                    "drift_reports/",
                    "baseline/",
                    "Tudo gerado no preprocess",
                    "  e no drift check",
                ],
            ),
            (
                "models/evaluation + deployments/",
                [
                    "evaluation_report.json",
                    "*.csv.out (scores batch)",
                    "predictions_enriched/",
                    "deployment_report.json",
                    "realtime-serving/*/…",
                    "(artefato de serving)",
                ],
            ),
        ],
        subtitle="Models Registry ficam de fora deste capítulo — foco em dados e evidências",
    )

    content(
        prs,
        "Artefatos de features (CSV + Parquet + tabelas Glue)",
        [
            {"text": "Gerador", "bold": True, "color": TEAL, "plain": True, "size": 16},
            "feature_engineering.py (Glue preprocess) a partir de campaign_results / transformed/",
            {"text": "CSV para SageMaker (sem header)", "bold": True, "color": TEAL, "plain": True, "size": 16},
            "features/train/ · validation/ · test/ — consumo direto do Training/Transform XGBoost",
            "features/test_transform/ — mesmas features sem coluna y (batch inference)",
            "features/baseline/ — CSV com header (monitoramento / inspeção)",
            {"text": "Parquet + catálogo", "bold": True, "color": TEAL, "plain": True, "size": 16},
            "features/train_parquet|validation_parquet|test_parquet/",
            "Tabelas Glue: {CATALOG_DB}.features_train|features_validation|features_test",
            {"text": "Split", "bold": True, "color": MUTED, "plain": True, "size": 16},
            "70% / 15% / 15% com seed fixa — reprodutibilidade entre runs",
        ],
    )

    content(
        prs,
        "Artefato: drift_reference/ (baseline do gate)",
        [
            {"text": "Caminho", "bold": True, "color": TEAL, "plain": True, "size": 16},
            "…/mlops_pipeline/data/bank_mktg/features/drift_reference/  (Parquet)",
            {"text": "Quando nasce", "bold": True, "color": TEAL, "plain": True, "size": 16},
            "No preprocess: snapshot da SOURCE_PATH (CSV bruto) no momento em que o treino leu a fonte",
            {"text": "Para que serve", "bold": True, "color": TEAL, "plain": True, "size": 16},
            "Referência do drift_check: compara fonte ao vivo vs este snapshot (PSI / share-delta)",
            {"text": "Se estiver ausente", "bold": True, "color": ACCENT, "plain": True, "size": 16},
            "result.json marca drift_detected=true / reason=no_reference → permite bootstrap do 1º treino",
            {"text": "Não confundir com", "bold": True, "color": MUTED, "plain": True, "size": 16},
            "features/baseline/ (CSV com header do split de treino) — propósito de inspeção, não o gate GHA",
        ],
    )

    content(
        prs,
        "Artefato: drift_reports/**/result.json",
        [
            {"text": "Caminhos", "bold": True, "color": TEAL, "plain": True, "size": 16},
            "…/drift_reports/<ts_nodash>/result.json  e  …/drift_reports/latest/result.json",
            {"text": "Gerador", "bold": True, "color": TEAL, "plain": True, "size": 16},
            "glue_data_drift_check.py via DAG drift_check_pipeline (job stack-data-drift-{proj.name})",
            {"text": "Conteúdo típico", "bold": True, "color": TEAL, "plain": True, "size": 16},
            "drift_detected (bool) · reason · psi_threshold · reference_path · current_path · columns[]",
            "Por coluna: metric_type (psi|share_delta), metric_value, threshold, drifted",
            {"text": "Consumidor", "bold": True, "color": TEAL, "plain": True, "size": 16},
            "mlops_helper.py read-drift-result → GITHUB_OUTPUT → decide se chama training_pipeline",
            {"text": "Importante", "bold": True, "color": ACCENT, "plain": True, "size": 16},
            "É evidência de dados, não de modelo — gate de retreino baseado em mudança da fonte",
        ],
    )

    content(
        prs,
        "Artefatos de avaliação (sem persistir Model)",
        [
            {"text": "Batch Transform output", "bold": True, "color": TEAL, "plain": True, "size": 16},
            "…/models/evaluation/<run>/…/*.csv.out — scores brutos por linha",
            {"text": "evaluation_report.json", "bold": True, "color": TEAL, "plain": True, "size": 16},
            "AUC, accuracy, F1, test_samples, training_job, model_data_url, enriched_predictions_uri",
            {"text": "predictions_enriched/ (Delta Lake)", "bold": True, "color": TEAL, "plain": True, "size": 16},
            "Tabela estável append-only: score + features (JSON) + model_name/version + inference_timestamp",
            "Permite análise histórica de predições batch sem depender do Model efêmero",
            {"text": "MLflow", "bold": True, "color": TEAL, "plain": True, "size": 16},
            "Experimento stack-experiments — params/metrics/tags do evaluate (e depois do deploy)",
            {"text": "Limpeza", "bold": True, "color": MUTED, "plain": True, "size": 16},
            "O Model stack-model-* usado no transform é apagado após o job — reports/Delta permanecem",
        ],
    )

    # ========== PROMOTE / SERVING ARTIFACTS ==========
    section(
        prs,
        "03",
        "Artefatos de Promote / serving",
        "O que o promote gera além do Model: packages staged, reports, metadata e infra de trigger.",
    )

    content(
        prs,
        "Artefatos de deploy (não são “My models”)",
        [
            {"text": "deployment_report.json", "bold": True, "color": TEAL, "plain": True, "size": 16},
            "Gravado pelo Processing de deploy em …/deployments/ — status, ARNs, git_commit, timestamps",
            {"text": "realtime-serving/<nome>/model.tar.gz", "bold": True, "color": TEAL, "plain": True, "size": 16},
            "Pacote de serving: booster + code/inference.py + feature_schema.py + serving_meta.json",
            "Garante resposta enriquecida (score, features nomeadas, model, timestamp)",
            {"text": "Stage cross-project", "bold": True, "color": TEAL, "plain": True, "size": 16},
            "Promote copia artefato para o S3 shared de test/prod (MODEL_DATA_URL_OVERRIDE)",
            "Evita o role do estágio ler o bucket de origem sem permissão",
            {"text": "CustomerMetadataProperties no package", "bold": True, "color": TEAL, "plain": True, "size": 16},
            "last_deployed_stage / deployed_to_*_at — trilha de promoção e anti-loop do EventBridge",
            {"text": "Gates humanos", "bold": True, "color": MUTED, "plain": True, "size": 16},
            "GitHub Issues abertas pelo promote (aprovação test/prod) — artefato de processo/auditoria",
        ],
    )

    content(
        prs,
        "Artefatos de infraestrutura (setup-infra)",
        [
            {"text": "Provisionados por e2e-mlops-pipeline (setup_infra=true)", "bold": True, "color": TEAL, "plain": True, "size": 16},
            "EventBridge rule (default stack-model-approved) — Model Package State Change / Approved",
            "Lambda (default stack-airflow-deploy-trigger) — dispara repository_dispatch model-approved",
            "IAM role da Lambda + permissões Secrets Manager (token GitHub)",
            {"text": "Não são dados", "bold": True, "color": MUTED, "plain": True, "size": 16},
            "Mas são artefatos operacionais necessários para o promote automático após aprovação",
            {"text": "Idempotência", "bold": True, "color": TEAL, "plain": True, "size": 16},
            "setup-mlops-infra.sh atualiza código/config existentes — reexecutar após mudança da Lambda",
        ],
    )

    # ========== LINHAGEM ==========
    section(
        prs,
        "04",
        "Linhagem ponta a ponta",
        "Como os artefatos se encadeiam — do CSV ao endpoint — sem depender dos Models da UI.",
    )

    content(
        prs,
        "Cadeia de dependências (artefato → artefato)",
        [
            "stack-sample.csv → staging/ → transformed/ ⇔ campaign_results",
            "transformed/ → quality_reports/ ⇔ dq_reports  (gate de qualidade DataOps)",
            "campaign_results → features/* + drift_reference/ + tabelas features_*",
            "CSV vivo + drift_reference/ → drift_reports/result.json → decide treino GHA",
            "features/train|val → job de treino → artefato S3 do booster (registrado no Registry)",
            "features/test + booster → evaluation_report.json + predictions_enriched + MLflow",
            "Package aprovado → realtime-serving tar.gz + deployment_report + endpoint",
            {"text": "Conclusão: a verdade operacional está nesses arquivos/tabelas; Models soltos são runtime", "bold": True, "color": ACCENT},
        ],
    )

    two_col(
        prs,
        "Quem lê o quê em cada estágio",
        "DEV",
        [
            "DataOps atualiza campaign_results + dq_reports",
            "MLOps treina/avalia/drift sobre esses dados",
            "Endpoint stack-prediction-dev serve o package",
            "Único estágio com linhagem de dados completa neste exemplo",
        ],
        "TEST / PROD",
        [
            "Recebem artefato de modelo (stage S3) + DAGs de deploy",
            "Endpoint NÃO lê campaign_results de dev em runtime",
            "Não há promote automático de dataset DataOps aqui",
            "Modelo foi aprendido em dados de dev; serving é só score",
        ],
    )

    # resumo inventário
    content(
        prs,
        "Checklist de inventário (validar no projeto)",
        [
            {"text": "DataOps", "bold": True, "color": TEAL, "plain": True, "size": 16},
            "□ staging/  □ transformed/  □ campaign_results  □ quality_reports/  □ dq_reports  □ athena_results/",
            {"text": "MLOps dados", "bold": True, "color": TEAL, "plain": True, "size": 16},
            "□ features/train|val|test  □ *_parquet + features_*  □ drift_reference/  □ drift_reports/latest/result.json",
            {"text": "MLOps evidência", "bold": True, "color": TEAL, "plain": True, "size": 16},
            "□ evaluation_report.json  □ predictions_enriched/  □ runs MLflow stack-experiments",
            {"text": "Promote / serving", "bold": True, "color": TEAL, "plain": True, "size": 16},
            "□ deployments/deployment_report.json  □ realtime-serving/…/model.tar.gz  □ metadata no package",
            "□ EventBridge + Lambda  □ Issues de aprovação (test/prod)",
        ],
        subtitle="Models Registry / endpoints existem, mas este checklist é o que sustenta auditoria e linhagem",
    )

    content(
        prs,
        "Mensagens-chave para o time",
        [
            {"text": "1. campaign_results + dq_reports são os contratos DataOps visíveis no Athena", "bold": True},
            {"text": "2. features/* e drift_reference/ são os contratos de treino/gate — sem eles o MLOps não fecha o ciclo", "bold": True},
            {"text": "3. evaluation_report + predictions_enriched + MLflow são a evidência de qualidade do modelo", "bold": True},
            {"text": "4. realtime-serving + deployment_report + metadata são a evidência de promoção", "bold": True},
            {"text": "5. test/prod servem o modelo; não espelham automaticamente o lake DataOps de dev", "bold": True, "color": ACCENT},
        ],
    )

    title_slide(
        prs,
        "Perguntas?",
        "Inventário de artefatos: S3 Shared · Athena · MLflow · reports · infra de trigger",
        "Arquivo: CICD-DataOps-MLOps-Promote-PT-v2.pptx",
    )

    for path in (OUT_DOCS, OUT_STACK):
        path.parent.mkdir(parents=True, exist_ok=True)
        prs.save(path)
        print(f"Wrote {path}")


if __name__ == "__main__":
    build()
