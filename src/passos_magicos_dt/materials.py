from __future__ import annotations

import json
from pathlib import Path
from textwrap import dedent

import nbformat as nbf

from .analysis import AnalyticsArtifacts
from .config import (
    ANALYTICS_DIR,
    DATA_NOTEBOOK_NAME,
    LOGS_DIR,
    NOTEBOOKS_DIR,
    PAINEL_NOTEBOOK_NAME,
    PAINEL_PAYLOAD_NAME,
    QNA_REPORT_NAME,
    RUNNER_NOTEBOOK_NAME,
)


def ensure_material_dirs() -> None:
    for diretorio in (ANALYTICS_DIR, NOTEBOOKS_DIR, LOGS_DIR):
        diretorio.mkdir(parents=True, exist_ok=True)


def write_qna_report(artefatos_analiticos: AnalyticsArtifacts) -> Path:
    ensure_material_dirs()
    linhas = [
        "# Questões analíticas",
        "",
        "Documento com respostas, número-chave, leitura gerencial e ação recomendada.",
        "",
    ]

    for item in artefatos_analiticos.qna_items:
        linhas.extend(
            [
                f"## Pergunta {item['id']} - {item['title']}",
                "",
                f"**Resposta curta:** {item['answer']}",
                "",
                f"**Número-chave:** `{item['key_number']}`",
                "",
                f"**Leitura gerencial:** {item['managerial_takeaway']}",
                "",
                f"**Ação recomendada:** {item['recommended_action']}",
                "",
                f"**Figura:** `{item['figure']}`",
                "",
            ]
        )

    linhas.extend(["# Análises complementares", ""])
    for item in artefatos_analiticos.extra_items:
        linhas.extend(
            [
                f"## {item['title']}",
                "",
                f"**Insight:** {item['answer']}",
                "",
                f"**Número-chave:** `{item['key_number']}`",
                "",
                f"**Leitura gerencial:** {item['managerial_takeaway']}",
                "",
                f"**Ação recomendada:** {item['recommended_action']}",
                "",
                f"**Figura:** `{item['figure']}`",
                "",
            ]
        )

    caminho_saida = ANALYTICS_DIR / QNA_REPORT_NAME
    caminho_saida.write_text("\n".join(linhas), encoding="utf-8")
    return caminho_saida


def write_painel_payload(artefatos_analiticos: AnalyticsArtifacts) -> Path:
    ensure_material_dirs()
    carga_painel = {
        "resumo_painel": artefatos_analiticos.resumo_painel,
        "qna_items": artefatos_analiticos.qna_items,
        "extra_items": artefatos_analiticos.extra_items,
    }
    caminho_saida = ANALYTICS_DIR / PAINEL_PAYLOAD_NAME
    caminho_saida.write_text(
        json.dumps(carga_painel, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    return caminho_saida


def _codigo_ambiente_notebook(*, imports_modulo: str = "") -> str:
    return dedent(
        f"""
        import sys
        from pathlib import Path


        DIRETORIO_NOTEBOOKS = Path.cwd()
        if not (DIRETORIO_NOTEBOOKS / "ambiente_notebook.py").exists():
            candidato = DIRETORIO_NOTEBOOKS / "notebooks"
            if candidato.exists():
                DIRETORIO_NOTEBOOKS = candidato
        if str(DIRETORIO_NOTEBOOKS) not in sys.path:
            sys.path.insert(0, str(DIRETORIO_NOTEBOOKS))

        from ambiente_notebook import adicionar_raiz_no_syspath{imports_modulo}

        DIRETORIO_RAIZ = adicionar_raiz_no_syspath()
        """
    ).strip()


def _data_preparation_notebook_cells() -> list[nbf.NotebookNode]:
    return [
        nbf.v4.new_markdown_cell(
            dedent(
                """
                # Tratamento dos dados e geração das bases

                Use este notebook para:

                1. localizar os arquivos de origem;
                2. montar a base analítica por aluno e ano;
                3. gerar a base temporal usada na modelagem;
                4. treinar o modelo e salvar os artefatos.
                """
            ).strip()
        ),
        nbf.v4.new_code_cell(
            dedent(
                """
                """
            ).strip()
            + "\n\n"
            + _codigo_ambiente_notebook(imports_modulo=", caminho_relativo_projeto")
            + "\n\n"
            + dedent(
                """

                import pandas as pd
                from IPython.display import display

                from src.passos_magicos_dt.analysis import build_analytics
                from src.passos_magicos_dt.data import prepare_datasets, resolve_legacy_csv_path
                from src.passos_magicos_dt.materials import write_painel_payload, write_qna_report
                from src.passos_magicos_dt.modeling import save_training_artifacts, train_temporal_model
                """
            ).strip()
        ),
        nbf.v4.new_markdown_cell("## 1. Conferir as fontes"),
        nbf.v4.new_code_cell(
            dedent(
                """
                diretorio_raiz = DIRETORIO_RAIZ
                pacote_dados = prepare_datasets(root=diretorio_raiz)
                arquivo_legado = resolve_legacy_csv_path(root=diretorio_raiz)

                quadro_caminhos = pd.DataFrame(
                    [
                        {"fonte": "Excel oficial", "caminho": caminho_relativo_projeto(pacote_dados.caminho_excel)},
                        {"fonte": "CSV legado", "caminho": caminho_relativo_projeto(arquivo_legado)},
                    ]
                )
                quadro_caminhos
                """
            ).strip()
        ),
        nbf.v4.new_markdown_cell("## 2. Conferir a base analítica"),
        nbf.v4.new_code_cell(
            dedent(
                """
                resumo_base = pd.DataFrame(
                    {
                        "registros": [len(pacote_dados.base_analitica)],
                        "alunos_unicos": [pacote_dados.base_analitica["ra"].nunique()],
                        "anos": [", ".join(map(str, sorted(pacote_dados.base_analitica["ano_referencia"].unique())))],
                    }
                )
                display(resumo_base)
                pacote_dados.base_analitica.head()
                """
            ).strip()
        ),
        nbf.v4.new_code_cell(
            dedent(
                """
                percentual_ausencias = (
                    pacote_dados.base_analitica.isna()
                    .mean()
                    .sort_values(ascending=False)
                    .head(15)
                    .rename("pct_missing")
                    .reset_index()
                    .rename(columns={"index": "coluna"})
                )
                percentual_ausencias
                """
            ).strip()
        ),
        nbf.v4.new_markdown_cell("## 3. Conferir a base de pares"),
        nbf.v4.new_code_cell(
            dedent(
                """
                resumo_pares = pd.DataFrame(
                    {
                        "registros": [len(pacote_dados.base_pares)],
                        "anos_referencia": [", ".join(map(str, sorted(pacote_dados.base_pares["ano_referencia"].unique())))],
                        "taxa_risco_proximo_ano": [pacote_dados.base_pares["risco_proximo_ano"].mean()],
                    }
                )
                display(resumo_pares)
                pacote_dados.base_pares[["ra", "ano_referencia", "ano_alvo", "risco_atual", "risco_proximo_ano", "delta_inde"]].head()
                """
            ).strip()
        ),
        nbf.v4.new_markdown_cell("## 4. Treinar e salvar os artefatos"),
        nbf.v4.new_code_cell(
            dedent(
                """
                artefatos_treinamento = train_temporal_model(pacote_dados.base_pares)
                save_training_artifacts(artefatos_treinamento)
                artefatos_treinamento.metrics_holdout
                """
            ).strip()
        ),
        nbf.v4.new_code_cell(
            dedent(
                """
                artefatos_analiticos = build_analytics(
                    pacote_dados.base_analitica,
                    pacote_dados.base_pares,
                    artefatos_treinamento,
                )
                write_qna_report(artefatos_analiticos)
                write_painel_payload(artefatos_analiticos)
                artefatos_treinamento.feature_importance.head(12)
                """
            ).strip()
        ),
        nbf.v4.new_markdown_cell("## 5. Saídas geradas"),
        nbf.v4.new_markdown_cell(
            "Ao final da execução, o projeto atualiza `data/processed/`, `artifacts/model/` e `artifacts/analytics/`."
        ),
    ]


def _painel_analitico_notebook_cells() -> list[nbf.NotebookNode]:
    return [
        nbf.v4.new_markdown_cell(
            dedent(
                """
                # Painel analítico

                Este notebook reúne o resumo gerencial, as respostas analíticas e as métricas do modelo já geradas pelo pipeline.
                """
            ).strip()
        ),
        nbf.v4.new_code_cell(
            dedent(
                """
                """
            ).strip()
            + "\n\n"
            + _codigo_ambiente_notebook(imports_modulo=", resolver_caminho_projeto")
            + "\n\n"
            + dedent(
                """

                import json

                import pandas as pd
                from IPython.display import Image, Markdown, display

                from src.passos_magicos_dt.config import ANALYTICS_DIR, MODEL_DIR
                """
            ).strip()
        ),
        nbf.v4.new_code_cell(
            dedent(
                """
                carga_painel = json.loads((ANALYTICS_DIR / "painel_analitico.json").read_text(encoding="utf-8"))
                resumo_painel = pd.DataFrame(carga_painel["resumo_painel"].items(), columns=["indicador", "valor"])
                resumo_painel
                """
            ).strip()
        ),
        nbf.v4.new_markdown_cell("## 1. Perguntas analíticas"),
        nbf.v4.new_code_cell(
            dedent(
                """
                for item in carga_painel["qna_items"]:
                    display(Markdown(f"## Pergunta {item['id']} - {item['title']}"))
                    linhas = [
                        f"**Resposta:** {item['answer']}",
                        f"**Número-chave:** `{item['key_number']}`",
                        f"**Leitura gerencial:** {item['managerial_takeaway']}",
                        f"**Ação recomendada:** {item['recommended_action']}",
                    ]
                    display(Markdown(("  " + chr(10)).join(linhas)))
                    display(Image(filename=str(resolver_caminho_projeto(item["figure"]))))
                """
            ).strip()
        ),
        nbf.v4.new_markdown_cell("## 2. Leituras complementares"),
        nbf.v4.new_code_cell(
            dedent(
                """
                for item in carga_painel["extra_items"]:
                    display(Markdown(f"## {item['title']}"))
                    linhas = [
                        f"**Insight:** {item['answer']}",
                        f"**Número-chave:** `{item['key_number']}`",
                        f"**Leitura gerencial:** {item['managerial_takeaway']}",
                        f"**Ação recomendada:** {item['recommended_action']}",
                    ]
                    display(Markdown(("  " + chr(10)).join(linhas)))
                    display(Image(filename=str(resolver_caminho_projeto(item["figure"]))))
                """
            ).strip()
        ),
        nbf.v4.new_markdown_cell("## 3. Métricas do modelo"),
        nbf.v4.new_code_cell(
            dedent(
                """
                metricas_holdout = pd.read_csv(MODEL_DIR / "metrics_holdout.csv")
                importancia_variaveis = pd.read_csv(MODEL_DIR / "feature_importance.csv").head(15)
                display(metricas_holdout)
                importancia_variaveis
                """
            ).strip()
        ),
        nbf.v4.new_markdown_cell(
            "Os mesmos arquivos lidos aqui alimentam a página analítica do app."
        ),
    ]


def _runner_notebook_cells() -> list[nbf.NotebookNode]:
    return [
        nbf.v4.new_markdown_cell(
            dedent(
                """
                # Executar pipeline e subir o Streamlit

                Use este notebook quando quiser rodar o fluxo inteiro no ambiente local:

                1. executar os notebooks 01 e 02;
                2. conferir os artefatos principais;
                3. iniciar o Streamlit.
                """
            ).strip()
        ),
        nbf.v4.new_code_cell(
            dedent(
                """
                """
            ).strip()
            + "\n\n"
            + _codigo_ambiente_notebook()
            + "\n\n"
            + dedent(
                """
                import subprocess
                import sys
                import time
                import urllib.request
                import webbrowser

                import nbformat as nbf
                import pandas as pd
                from nbconvert.preprocessors import ExecutePreprocessor

                NOTEBOOKS_FLUXO = [
                    DIRETORIO_RAIZ / "notebooks" / "01_tratamento_e_bases_modelo.ipynb",
                    DIRETORIO_RAIZ / "notebooks" / "02_painel_analitico.ipynb",
                ]

                PROCESSO_STREAMLIT = None
                """
            ).strip()
        ),
        nbf.v4.new_code_cell(
            dedent(
                """
                def executar_notebook(caminho_notebook: Path) -> Path:
                    with caminho_notebook.open("r", encoding="utf-8") as arquivo:
                        notebook = nbf.read(arquivo, as_version=4)

                    executor = ExecutePreprocessor(timeout=900, kernel_name="python3")
                    executor.preprocess(notebook, {"metadata": {"path": str(caminho_notebook.parent)}})

                    with caminho_notebook.open("w", encoding="utf-8") as arquivo:
                        nbf.write(notebook, arquivo)

                    return caminho_notebook


                def executar_fluxo_notebooks() -> pd.DataFrame:
                    linhas = []
                    for caminho_notebook in NOTEBOOKS_FLUXO:
                        inicio_execucao = time.time()
                        executar_notebook(caminho_notebook)
                        linhas.append(
                            {
                                "notebook": caminho_notebook.name,
                                "status": "executado",
                                "duracao_seg": round(time.time() - inicio_execucao, 1),
                            }
                        )
                    return pd.DataFrame(linhas)


                def aguardar_streamlit(url_base: str, tentativas: int = 20, pausa_seg: float = 1.0) -> bool:
                    for _ in range(tentativas):
                        try:
                            with urllib.request.urlopen(url_base, timeout=2):
                                return True
                        except Exception:
                            time.sleep(pausa_seg)
                    return False


                def iniciar_streamlit(
                    porta: int = 8501,
                    modo_headless: bool = True,
                    abrir_navegador: bool = True,
                ) -> int:
                    global PROCESSO_STREAMLIT
                    if PROCESSO_STREAMLIT is not None and PROCESSO_STREAMLIT.poll() is None:
                        print(f"Streamlit já está rodando no PID {PROCESSO_STREAMLIT.pid}.")
                        return PROCESSO_STREAMLIT.pid

                    caminho_log = DIRETORIO_RAIZ / "artifacts" / "logs" / "streamlit_runner.log"
                    caminho_log.parent.mkdir(parents=True, exist_ok=True)
                    arquivo_log = caminho_log.open("w", encoding="utf-8")
                    comando = [
                        sys.executable,
                        "-m",
                        "streamlit",
                        "run",
                        "app.py",
                        "--server.port",
                        str(porta),
                        "--server.headless",
                        "true" if modo_headless else "false",
                    ]
                    PROCESSO_STREAMLIT = subprocess.Popen(
                        comando,
                        cwd=str(DIRETORIO_RAIZ),
                        stdout=arquivo_log,
                        stderr=subprocess.STDOUT,
                    )

                    url = f"http://localhost:{porta}"
                    if aguardar_streamlit(url):
                        if abrir_navegador:
                            webbrowser.open_new_tab(url)
                        print(f"Streamlit iniciado. PID={PROCESSO_STREAMLIT.pid} | URL={url}")
                    else:
                        print("O Streamlit foi iniciado, mas a URL não respondeu no tempo esperado.")

                    print(f"Log: {caminho_log}")
                    return PROCESSO_STREAMLIT.pid


                def encerrar_streamlit() -> None:
                    global PROCESSO_STREAMLIT
                    if PROCESSO_STREAMLIT is None or PROCESSO_STREAMLIT.poll() is not None:
                        print("Não há processo ativo do Streamlit.")
                        return
                    PROCESSO_STREAMLIT.terminate()
                    PROCESSO_STREAMLIT.wait(timeout=15)
                    print("Streamlit finalizado.")
                    PROCESSO_STREAMLIT = None
                """
            ).strip()
        ),
        nbf.v4.new_markdown_cell("## 1. Executar os notebooks"),
        nbf.v4.new_code_cell(
            dedent(
                """
                relatorio_execucao = executar_fluxo_notebooks()
                relatorio_execucao
                """
            ).strip()
        ),
        nbf.v4.new_markdown_cell("## 2. Conferir os artefatos"),
        nbf.v4.new_code_cell(
            dedent(
                """
                artefatos_principais = [
                    DIRETORIO_RAIZ / "data" / "processed" / "base_analitica.parquet",
                    DIRETORIO_RAIZ / "data" / "processed" / "base_modelagem_proximo_ano.parquet",
                    DIRETORIO_RAIZ / "artifacts" / "model" / "model_pipeline.joblib",
                    DIRETORIO_RAIZ / "artifacts" / "analytics" / "painel_analitico.json",
                ]

                quadro_validacao = pd.DataFrame(
                    [
                        {
                            "arquivo": caminho.name,
                            "existe": caminho.exists(),
                            "caminho": str(caminho),
                        }
                        for caminho in artefatos_principais
                    ]
                )
                quadro_validacao
                """
            ).strip()
        ),
        nbf.v4.new_markdown_cell(
            "## 3. Iniciar o Streamlit\n\nA célula abaixo inicia o app local e tenta abrir o navegador quando a URL estiver pronta."
        ),
        nbf.v4.new_code_cell(
            dedent(
                """
                iniciar_streamlit(porta=8501, modo_headless=True, abrir_navegador=True)
                """
            ).strip()
        ),
        nbf.v4.new_markdown_cell("## 4. Encerrar o Streamlit"),
        nbf.v4.new_code_cell(
            dedent(
                """
                # Execute esta célula apenas quando quiser encerrar o app.
                # encerrar_streamlit()
                """
            ).strip()
        ),
    ]


def _write_notebook(caminho: Path, celulas: list[nbf.NotebookNode]) -> Path:
    ensure_material_dirs()
    notebook = nbf.v4.new_notebook()
    notebook.cells = celulas
    caminho.write_text(nbf.writes(notebook), encoding="utf-8")
    return caminho


def write_data_preparation_notebook() -> Path:
    return _write_notebook(
        NOTEBOOKS_DIR / DATA_NOTEBOOK_NAME,
        _data_preparation_notebook_cells(),
    )


def write_painel_analitico_notebook() -> Path:
    return _write_notebook(
        NOTEBOOKS_DIR / PAINEL_NOTEBOOK_NAME,
        _painel_analitico_notebook_cells(),
    )


def write_runner_notebook() -> Path:
    return _write_notebook(
        NOTEBOOKS_DIR / RUNNER_NOTEBOOK_NAME,
        _runner_notebook_cells(),
    )
